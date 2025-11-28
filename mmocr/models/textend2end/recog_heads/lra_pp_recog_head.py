import sys
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from mmdet.core import multi_apply
from mmdet.models.builder import HEADS
from mmocr.models.builder import build_decoder, build_convertor, build_encoder
from mmocr.models.textdet.dense_heads import HeadMixin
from ..utils.tps_align import TPSAlign
from mmocr.utils.tps_util import TPS

class CTCLoss(nn.Module):

    def __init__(self, blank, zero_infinity=True, **kwargs):
        super(CTCLoss, self).__init__()
        self.loss_func = nn.CTCLoss(blank=blank,
                                    reduction='mean',
                                    zero_infinity=zero_infinity)

    def forward(self, predicts, label, label_length):

        batch_size = predicts.size(0)
        predicts = predicts.log_softmax(2)
        predicts = predicts.permute(1, 0, 2)
        preds_lengths = torch.tensor([predicts.size(0)] * batch_size,
                                     dtype=torch.long)
        loss = self.loss_func(predicts, label, preds_lengths, label_length)

        return loss


    
@HEADS.register_module()
class LRANet_PP_RecogHead(nn.Module,HeadMixin):

    def __init__(self, recognizer,path_lra,num_coefficients=14,
                 sample_size=(32,100), convertor=None, from_p2=True,
                 image_size=(960,960),add_gt=True, max_num=100,
                 num_sample_per_ins=2,fiducial_dist='cross',
                 ):
        super(LRANet_PP_RecogHead, self).__init__()

        self.recognizer = build_encoder(recognizer)
        self.convertor = build_convertor(convertor)
        self.character = convertor['dict_list']
        self.pad_idx=recognizer['voc_size']
        self.blank_idx = recognizer['voc_size']
        self.unknown_idx=recognizer['voc_size']-1
        self.max_seq_len = self.convertor.max_seq_len
        self.num_sample_per_ins = num_sample_per_ins
        self.sample_size = sample_size
        self.fiducial_dist = fiducial_dist
        self.from_p2 = from_p2
        self.add_gt = add_gt
        self.max_num = max_num
        self.img_w, self.img_h = image_size
        self.tpsalign_gt = TPSAlign(14, grid_size=sample_size)
        self.tpsalign = TPSAlign(grid_size=sample_size, fiducial_dist=fiducial_dist)
        self.loss_recog = CTCLoss(blank=self.blank_idx)
        U_t = np.load(path_lra)['components_c']
        self.U_t = torch.from_numpy(U_t).cuda()

        # In lranet_targets.py, we set n = self.num_coefficients // 2.
        self.tps_decode = TPS(num_points=num_coefficients,grid_size=sample_size, fiducial_dist=fiducial_dist)

    def get_height(self,grids):

        p1 = grids[:, 0, :]
        p2 = grids[:, -self.sample_size[1], :]
        height = ((p1 - p2) ** 2).sum(dim=1).sqrt()
        return height

    def assign_grids_to_level(self,
            grids, min_level=2, max_level=4, canonical_box_size=96,
            canonical_level=3):

        eps = sys.float_info.epsilon
        box_sizes = self.get_height(grids)
        # Eqn.(1) in FPN paper
        level_assignments = torch.floor(
            canonical_level + torch.log2(box_sizes / canonical_box_size + eps)
        )
        level_assignments = torch.clamp(level_assignments, min=min_level, max=max_level)
        return level_assignments.to(torch.int64) - min_level


    def forward(self,feat_maps,  preds, p3_maps, p4_maps, p5_maps, gt_texts=None, polygons_area=None, lv_tps_coeffs=None,**kwargs):


        device = feat_maps[0].device
        gt_maps = [p3_maps, p4_maps, p5_maps]
        lv_img_tps_coeffs = list(zip(*lv_tps_coeffs))

        # gt_texts should be tensors
        gt_texts = [self.convertor.str2tensor(text.data) for text in gt_texts]
        max_num_polygon = max([len(p) for p in polygons_area])
        pad_polygon_areas = torch.zeros(len(polygons_area), max_num_polygon, device=device)
        pad_gt_texts = torch.zeros(len(gt_texts),max_num_polygon, self.max_seq_len, device=device, dtype=torch.long)
        for bi, po in enumerate(polygons_area):
            if len(po) == 0:
                continue
            pad_polygon_areas[bi, :len(po)] = torch.from_numpy(polygons_area[bi]).to(device)
            pad_gt_texts[bi, :len(po)] = gt_texts[bi].to(device)

        pad_polygon_areas = [pad_polygon_areas] * 3
        pad_gt_texts = [pad_gt_texts] * 3
        downsample_rates = [8,16,32]
        if preds is None:
            preds = [preds] * 3
            
        sample_grids, sample_gt_texts, sample_idxs, batch_index = multi_apply(self.sample_feature,feat_maps, preds, gt_maps, pad_polygon_areas,pad_gt_texts, downsample_rates)

        gt_grids, gt_gt_texts, gt_batch_index = multi_apply(self.sample_feature_gt, feat_maps, lv_img_tps_coeffs,
                                                      pad_gt_texts, downsample_rates)


        if self.add_gt:
            sample_grids += gt_grids
            sample_gt_texts+= gt_gt_texts
            batch_index+=gt_batch_index

        sample_grids = torch.cat(sample_grids, dim=0)
        sample_gt_texts = torch.cat(sample_gt_texts, dim=0).long()
        batch_index = torch.cat(batch_index, dim=0)

        grids_level = self.assign_grids_to_level(sample_grids)

        sample_grids[:, :, 0], sample_grids[:, :, 1] = sample_grids[:, :, 0] * 2 / self.img_w - 1, sample_grids[:, :,
                                                                                         1] * 2 / self.img_h - 1

        sample_aligned_feature = []
        sample_aligned_gt_texts = []
        for level in range(len(feat_maps)):
            assign_idx = grids_level == level
            level_features,_ = self.tpsalign(feat_maps[level],sample_grids[assign_idx] , batch_index[assign_idx], None)
            sample_aligned_feature.append(level_features)
            sample_aligned_gt_texts.append(sample_gt_texts[assign_idx])


        sample_aligned_feature = torch.cat(sample_aligned_feature,dim=0)
        sample_aligned_gt_texts = torch.cat(sample_aligned_gt_texts, dim=0)
        
        if sample_aligned_feature.shape[0] == 0:
            loss =  torch.tensor(0.0, device=device,requires_grad=True).float()
        else:
            if sample_aligned_feature.shape[0]> self.max_num:
                random_indices = torch.randperm(sample_aligned_feature.shape[0])[:self.max_num]
                sample_aligned_feature = sample_aligned_feature[random_indices,:,:,:]
                sample_aligned_gt_texts = sample_aligned_gt_texts[random_indices,:]
                
            pred = self.recognizer(sample_aligned_feature)
            target_lengths = (sample_aligned_gt_texts != self.pad_idx).sum(dim=1).long()
            loss = self.loss_recog(pred, sample_aligned_gt_texts, target_lengths)

        return {'loss_ce':loss}



    def sample_feature_gt(self, feature_map, tps_coeffs, gt_texts, downsample_rate):

        device = feature_map.device
        H, W = feature_map.shape[-2:]
        batch_size = feature_map.shape[0]
        assert batch_size == len(tps_coeffs)
        batch_idx = []
        tps_coeffs_filterd = []
        for i in range(batch_size):
            batch_idx = batch_idx + [i] * tps_coeffs[i].shape[0]
            if tps_coeffs[i].shape[0] > 0:
                tps_coeffs_filterd.append(tps_coeffs[i])
        grids = torch.zeros(0, self.sample_size[0] * self.sample_size[1], 2, device=device)
        aligned_texts = torch.zeros(0, self.max_seq_len, device=device)
        batch_idx = torch.zeros(0, device=device)
        if len(batch_idx) > 0:
            tps_coeffs = torch.from_numpy(np.concatenate(tps_coeffs_filterd, 0)).to(device)
            batch_idx = torch.tensor(batch_idx, dtype=torch.long, device=device)
            ins_idx = tps_coeffs[:, 0].long() - 1
            tps_coeffs = tps_coeffs[:, 1:].view(tps_coeffs.shape[0], -1, 2).float()
            grids = self.tpsalign_gt.tps2grid(tps_coeffs) * downsample_rate
            aligned_texts = gt_texts[batch_idx, ins_idx]
        return grids, aligned_texts, batch_idx



        
    def sample_feature(self,feature_map, pred, gt_map, areas, gt_texts, downsample_rate):
        device = feature_map.device
        gt_map = gt_map.permute(0, 2, 3, 1).contiguous()
        train_mask = gt_map[:, :, :, 2:3].view(-1)

        tr_mask_idx = gt_map[:, :, :, :1].long()
        tcl_mask = gt_map[:, :, :, 1:2].view(-1)
        tr_mask = (tr_mask_idx != 0).view(-1)
        batch_size, H, W, _ = tr_mask_idx.shape
        ys, xs = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
        mesh_grid = torch.stack((xs, ys), dim=-1).long().to(device)
        mesh_grid = mesh_grid.repeat(batch_size, 1, 1, 1).view(-1, 2)
        batch_idx = torch.arange(0, batch_size)[:, None, None].repeat(1, H, W).to(device)
        batch_idx = batch_idx.view(-1)
        tr_mask_idx = tr_mask_idx.view(-1) - 1
        reg_pred = pred[1].permute(0, 2, 3, 1).contiguous()
        lra_map = reg_pred.reshape(-1, reg_pred.shape[-1])

        tr_train_mask = ((train_mask * tr_mask) > 0).float()
        pos_idx = torch.where(tr_train_mask > 0)[0]
        num_pos = tr_train_mask.sum().item()
        sample_grids = torch.zeros(0, self.sample_size[0]*self.sample_size[1],2,device=device)
        sample_aligned_gt_texts = torch.zeros(0,self.max_seq_len, device=device)
        sample_idx = []
        if num_pos > 0:
            batch_idx = batch_idx[pos_idx]
            tr_mask_idx = tr_mask_idx[pos_idx]
            center_weight = (0.5*tr_mask[pos_idx].float() +
                      0.5*tcl_mask[pos_idx].float())
            center_weight = center_weight.contiguous()
            weight =  center_weight
            sample_idx = self.sample_per_instance(batch_size, batch_idx, tr_mask_idx, weight, self.num_sample_per_ins)

            sample_pos_idx = pos_idx[sample_idx]
            batch_idx = batch_idx[sample_idx]
            tr_mask_idx = tr_mask_idx[sample_idx]
            lra_map = lra_map[sample_pos_idx]
            mesh_grid = mesh_grid[sample_pos_idx]

            polygons = torch.matmul(lra_map,self.U_t)
            polygons = polygons.reshape(polygons.shape[0],-1,2)
            tps_c = self.tps_decode.inverse_P_border(polygons)
            sample_grids = self.tps_decode.build_P_grid(tps_c)
            sample_grids = (sample_grids + mesh_grid[:,None,:]) * downsample_rate
            sample_gt_texts = gt_texts[batch_idx, tr_mask_idx]
            sample_aligned_gt_texts = sample_gt_texts
        else:
            batch_idx = torch.zeros(0, device=device)

        return sample_grids, sample_aligned_gt_texts, sample_idx,batch_idx


    def sample_per_instance(self,batch_size, batch_idx, tr_mask_idx, weight, n = 2):
        max_n_ins = tr_mask_idx.max().item() + 1
        index_all = batch_idx * max_n_ins + tr_mask_idx
        sample_index = []
        for i in range(batch_size * (max_n_ins)):
            index_instance = torch.where(index_all == i)[0]
            if len(index_instance) > 0:
                if len(index_instance) > n:
                    sample_index_instance = torch.multinomial(weight[index_instance] + 1e-3, n, replacement=False)
                else:
                    sample_index_instance = torch.multinomial(weight[index_instance]+1e-3, n, replacement=True)
                sample_index.append(index_instance[sample_index_instance])
        sample_index = torch.cat(sample_index)
        return sample_index



    def simple_test(self, feature_maps, det_results, img_metas, rescale):


        grids = [det_results['grids_results']]
        boundaries = [det_results['boundary_result']]
        
        lgrids = []
        lboundaries =[]
        for i in range(len(grids)):
            if len(grids[i]) > 0:
                lgrids.append(grids[i].view(grids[i].shape[0], self.sample_size[0]*self.sample_size[1],2) )
                lboundaries.append(np.array(boundaries[i] ))

        if len(lboundaries) == 0:
            return dict(boundary_result=[], grids_result=[], strs=[])
        boundaries = np.concatenate(lboundaries, axis=0)
        grids = torch.cat(lgrids,dim=0)
        assign_level = self.assign_grids_to_level(grids)
        # print(assign_level)
        h, w = feature_maps[0].shape[-2:]
        h, w = h*4, w*4
        if not self.from_p2:
            h*=2
            w*=2
        grids[:, :, 0] = grids[:, :,  0] * 2 / w - 1
        grids[:, :,  1] = grids[:, :, 1] * 2 / h - 1

        aligned_grids = []
        aligned_features = []
        aligned_boundaries = []
        for level in range(len(feature_maps)):
            assign_idx = assign_level == level
            grid = grids[assign_idx]
            aligned_grids.append(grid)
            aligned_feature,_ = self.tpsalign(feature_maps[level], grid.float(),
                                                torch.zeros(grid.shape[0], device=grid.device, dtype=torch.float32), None)
            aligned_features.append(aligned_feature)
            aligned_boundaries.append(boundaries[assign_idx.cpu().numpy()])

        boundaries = np.concatenate(aligned_boundaries)
        aligned_features = torch.cat(aligned_features)

        preds = self.recognizer(aligned_features)
        preds = F.softmax(preds, dim=2)
        preds = preds.detach().cpu().numpy()
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        texts = self.ctc_decode(preds_idx, preds_prob, is_remove_duplicate=True)


        strs = []
        for j, text in enumerate(texts):   
            strs.append(text[0])
            boundaries[j, -1] = boundaries[j, -1] * text[1]

        grids = torch.cat(aligned_grids, dim=0).cpu()
        boundaries = boundaries.tolist()


        if rescale:
            boundaries = self.resize_boundary(
                boundaries, 1.0 / img_metas[0]['scale_factor'])
            grids = self.resize_grid(grids, 1.0 / img_metas[0]['scale_factor'])
        

        results = dict(boundary_result=boundaries, grids_result=grids, strs=strs)
        return results


    def resize_grid(self, grids, scale_factor):
        assert isinstance(scale_factor, np.ndarray)
        assert scale_factor.shape[0] == 4
        assert grids.shape[-1] == 2

        for g in grids:
            sz = g.shape[0]
            g[:] = g[:] * np.tile(scale_factor[:2], (sz, 1))
        return grids

    

    def ctc_decode(self, text_index, text_prob=None, is_remove_duplicate=True):
        """convert text-index into text-label."""
        result_list = []
        ignored_tokens = [self.blank_idx,self.unknown_idx]
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            selection = np.ones(len(text_index[batch_idx]), dtype=bool)
            if is_remove_duplicate:
                selection[1:] = text_index[batch_idx][1:] != text_index[
                    batch_idx][:-1]
            for ignored_token in ignored_tokens:
                selection &= text_index[batch_idx] != ignored_token


            char_list = [
                self.character[text_id]
                for text_id in text_index[batch_idx][selection]
            ]
            if text_prob is not None:
                conf_list = text_prob[batch_idx][selection]
            else:
                conf_list = [1] * len(selection)
            if len(conf_list) == 0:
                conf_list = [0]

            text = ''.join(char_list)
            if len(text) > 0:
                conf_arr = np.array(conf_list)
                recog_score = np.exp(np.mean(np.log(conf_arr)))**5 * np.min(conf_arr)**0.25
            else:
                recog_score = 0
            result_list.append((text, recog_score))

        return result_list
    