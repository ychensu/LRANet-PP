import torch
from mmocr.models.textdet.detectors import FCENet
from mmdet.models.builder import build_head
from mmdet.models.builder import DETECTORS

@DETECTORS.register_module()
class LRANet_PP(FCENet):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 recog_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 show_score=False,
                 init_cfg=None,
                 from_p2=False):
        super(LRANet_PP, self).__init__(backbone, neck, bbox_head, train_cfg, test_cfg, pretrained, show_score, init_cfg)
        if recog_head is not None:
            self.recog_head = build_head(recog_head)
        else:
            self.recog_head = None
        self.from_p2 = from_p2

    def forward_train(self, img, img_metas, **kwargs):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys, see
                :class:`mmdet.datasets.pipelines.Collect`.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        x = self.extract_feat(img)
        preds = self.bbox_head(x[1:])
        losses = self.bbox_head.loss(preds, **kwargs)
        if self.recog_head is not None:
            recog_losses = self.recog_head(x[:-1], preds, **kwargs)
            losses.update(recog_losses)
        return losses

    def simple_test(self, img, img_metas, rescale=False):


        x = self.extract_feat(img)
        outs = self.bbox_head(x[1:])


        # early return to avoid post processing
        if torch.onnx.is_in_onnx_export():
            return outs

        if len(img_metas) > 1:
            boundaries = [
                self.bbox_head.get_boundary(*(outs[i].unsqueeze(0)),
                                            [img_metas[i]], rescale)
                for i in range(len(img_metas))
            ]

        else:
            boundaries = self.bbox_head.get_boundary(outs, img_metas, False if self.recog_head is not None else True)
            boundaries = [boundaries]

        if self.recog_head is not None:
            boundaries = self.recog_head.simple_test(x[:-1], boundaries[0], img_metas=img_metas, rescale=True)
            boundaries = [boundaries]

        return boundaries