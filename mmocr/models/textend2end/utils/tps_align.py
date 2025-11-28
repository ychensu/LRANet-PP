import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from mmocr.utils.tps_util import TPS
from .grid_sample_batch.grid_sample_batch import grid_sample_batch


class TPSAlign(nn.Module):

    def __init__(self, num_fiducial=8,tps_size=(0.25,1),grid_size=(32,100), fiducial_dist='edge'):
        super(TPSAlign, self).__init__()
        self.num_fiducial = num_fiducial
        self.grid_size=grid_size
        self.tps_size = tps_size
        self.fiducial_dist = fiducial_dist
        self.eps = 1e-5

        self.tps_decoder = TPS(num_fiducial, tps_size,grid_size=grid_size, fiducial_dist=fiducial_dist)


    def forward(self,feature_map, grids, batch_idx, texts):
        batch_size = feature_map.shape[0]
        grids = grids.detach()
        feats = grid_sample_batch(feature_map, grids.view(-1, self.grid_size[0], self.grid_size[1], 2), batch_idx.float(),0,0,None)
        aligned_texts = texts
        return feats, aligned_texts


    def tps2grid(self, tps_coeff):

        return self.tps_decoder.build_P_grid(tps_coeff)