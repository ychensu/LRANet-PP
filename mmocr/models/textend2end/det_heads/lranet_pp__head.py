import torch.nn as nn
import torch
from mmcv.cnn import normal_init, ConvModule
from mmcv.runner import BaseModule
from mmdet.core import multi_apply
from mmdet.models.builder import HEADS, build_loss
from mmocr.models.textend2end.postprocess import lra_decode
from mmocr.models.textdet.dense_heads.head_mixin import HeadMixin
from mmocr.utils.tps_util import TPS
from ..postprocess.lra_decode import  poly_nms
import numpy as np
import math

@HEADS.register_module()
class LRANet_PP_DetHead(HeadMixin, BaseModule):

    def __init__(self,
                 in_channels,
                 scales,
                 num_coefficients,
                 path_lra,
                 fiducial_dist='cross',
                 sample_size=(8,32),
                 loss=dict(type='LRALoss'),
                 score_thr=0.1,
                 nms_thr=0.1,
                 num_convs_stu=1,
                 num_convs_tea=4,
                 shift=0,
                 train_cfg=None,
                 test_cfg=None):

        super().__init__()
        assert isinstance(in_channels, int)

        self.downsample_ratio = 1.0
        self.in_channels = in_channels
        self.stu_channels = self.in_channels // 8
        self.num_convs_stu = num_convs_stu
        self.num_convs_tea = num_convs_tea
        self.scales = scales
        loss['steps'] = scales
        self.loss_module = build_loss(loss)
        self.score_thr = score_thr
        self.nms_thr = nms_thr
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.sample_size = sample_size
        self.shift = shift
        self.out_channels_reg = num_coefficients
        self.tps_decode = TPS(num_points=num_coefficients,grid_size=sample_size,fiducial_dist=fiducial_dist)
        U_t = np.load(path_lra)['components_c']
        self.U_t = torch.from_numpy(U_t).cuda()
        
        cls_convs = []
        reg_convs = []
        cls_convs_tea = []
        reg_convs_tea = []

        for i in range(self.num_convs_stu):
            chk_in = self.in_channels if i == 0 else self.stu_channels
            cls_convs.append(ConvModule(chk_in, self.stu_channels, kernel_size=1))
            reg_convs.append(ConvModule(chk_in, self.stu_channels, kernel_size=1))

        for j in range(self.num_convs_tea):
            cls_convs_tea.append(ConvModule(self.in_channels, self.in_channels, kernel_size=1))
            reg_convs_tea.append(ConvModule(self.in_channels, self.in_channels, kernel_size=1))

        self.cls_convs = nn.Sequential(*cls_convs)
        self.reg_convs = nn.Sequential(*reg_convs)
        self.cls_convs_tea = nn.Sequential(*cls_convs_tea)
        self.reg_convs_tea = nn.Sequential(*reg_convs_tea)

        self.out_conv_cls_dense = nn.Conv2d(
            self.stu_channels,
            1,
            kernel_size=3,
            stride=1,
            padding=1
            )
        
        self.out_conv_reg_dense = nn.Conv2d(
            self.stu_channels,
            self.out_channels_reg,
            kernel_size=3,
            stride=1,
            padding=1
            )

        self.out_conv_cls_sparse = nn.Conv2d(
            self.stu_channels,
            1,
            kernel_size=3,
            stride=1,
            padding=1
            )

        self.out_conv_reg_sparse = nn.Conv2d(
            self.stu_channels,
            self.out_channels_reg,
            kernel_size=3,
            stride=1,
            padding=1
            )

        self.out_conv_cls_sparse_tea = nn.Conv2d(
            self.in_channels,
            1,
            kernel_size=3,
            stride=1,
            padding=1)

        self.out_conv_reg_sparse_tea = nn.Conv2d(
            self.in_channels,
            self.out_channels_reg,
            kernel_size=3,
            stride=1,
            padding=1)


        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.out_conv_cls_sparse.bias, bias_value)
        torch.nn.init.constant_(self.out_conv_cls_sparse_tea.bias, bias_value)
        self.init_weights()


    def init_weights(self):
        normal_init(self.out_conv_cls_dense, mean=0, std=0.01)
        normal_init(self.out_conv_reg_dense, mean=0, std=0.01)
        normal_init(self.out_conv_reg_sparse, mean=0, std=0.01)
        normal_init(self.out_conv_reg_sparse_tea, mean=0, std=0.01)

    def forward(self, feats):

        if self.training:
            cls_dense, reg_dense, cls_sparse, reg_sparse, cls_sparse_tea, reg_sparse_tea = multi_apply(self.forward_single, feats)
            level_num = len(cls_dense)
            preds = [[cls_dense[i], reg_dense[i], cls_sparse[i], reg_sparse[i], cls_sparse_tea[i], reg_sparse_tea[i]] for i in range(level_num)]  
        else:
            cls_dense, cls_sparse, reg_sparse = multi_apply(self.forward_single, feats)
            level_num = len(cls_dense)
            preds = [[cls_dense[i], cls_sparse[i], reg_sparse[i]] for i in range(level_num)]                
        return preds

    def forward_single(self, x):
        if self.num_convs_stu > 0:
            x_cls = self.cls_convs(x)
            x_reg = self.reg_convs(x)
        else:
            x_cls = x
            x_reg = x   

        cls_predict_dense = self.out_conv_cls_dense(x_cls)
        cls_predict_sparse = self.out_conv_cls_sparse(x_cls)
        reg_predict_sparse = self.out_conv_reg_sparse(x_reg)

        if self.training:
            x_cls_tea = self.cls_convs_tea(x)
            x_reg_tea = self.reg_convs_tea(x)
            reg_predict_dense = self.out_conv_reg_dense(x_reg)
            cls_predict_sparse_tea = self.out_conv_cls_sparse_tea(x_cls_tea)
            reg_predict_sparse_tea = self.out_conv_reg_sparse_tea(x_reg_tea)

            return cls_predict_dense, reg_predict_dense, \
                cls_predict_sparse, reg_predict_sparse, \
                cls_predict_sparse_tea, reg_predict_sparse_tea   
        else:
            return cls_predict_dense, cls_predict_sparse, reg_predict_sparse 


    def get_boundary(self, score_maps, img_metas, rescale):

        assert len(score_maps) == len(self.scales)

        boundaries = []
        grids = []
        for idx, score_map in enumerate(score_maps):
            scale = self.scales[idx]
            boundary, grid = self._get_boundary_single(
                score_map, scale)
            boundaries = boundaries + boundary
            if len(grid) > 0:
                grids = grids + [grid*scale]

        # nms
        boundaries, keep_index = poly_nms(boundaries, self.nms_thr, with_index=True)


        if len(grids) > 0:
            grids = torch.cat(grids, dim=0)[keep_index]

        if rescale:
            boundaries = self.resize_boundary(
                boundaries, 1.0 / img_metas[0]['scale_factor'])

        results = dict(boundary_result=boundaries, grids_results=grids,scales=self.scales)
        return results



    def _get_boundary_single(self, score_map, scale):

        return lra_decode(
            preds=score_map,
            U_t=self.U_t,
            tps_decode=self.tps_decode,
            scale=scale,
            alpha=0.75,
            score_thr=0.1,
            shift=self.shift
        )

