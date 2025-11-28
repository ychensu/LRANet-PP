import torch
from torch import nn
import numpy as np


class TPS(nn.Module):
    """
    TPS encoder and decoder
    """

    def __init__(self, num_fiducial=8, fiducial_shape=(0.25,1),grid_size=(32,100),num_points=40, fiducial_dist="edge"):
        """Generate P_hat and inv_delta_C for later."""
        super().__init__()
        self.eps = 1e-6
        self.fiducial_height = fiducial_shape[0]
        self.fiducial_width = fiducial_shape[1]
        self.num_fiducial = num_fiducial
        assert fiducial_dist in ["edge", "cross", "center"]
        self.fiducial_dict = fiducial_dist
        if self.fiducial_dict == "edge":
            C = self._build_C_edge(num_fiducial)  # num_fiducial x 2
        elif self.fiducial_dict == "cross":
            C = self._build_C_cross()
        else:
            C = self._build_C_center()
        self.C = C
        # self.C = np.stack([C, C[:,[1,0]]])
        self.num_points = num_points
        P = self._build_P(num_points)
        self.P = P
        # self.P = np.stack([P, P[:,[1,0]]])
        # for multi-gpu, you need register buffer
        inv_delta_C = torch.tensor(self._build_inv_delta_C(self.num_fiducial, C)).float()
        self.register_buffer('inv_delta_C',inv_delta_C)

        P_hat =  torch.tensor(self._build_P_hat(self.num_fiducial, self.C,
                                     self.P)).float()  # n x num_fiducial+3
        # self.P_hat = P_hat # n x num_fiducial+3
        self.register_buffer("P_hat", P_hat)

        P_hat_inv = torch.linalg.pinv(P_hat)
        self.register_buffer("P_hat_inv", P_hat_inv)

        
        self.grid_size = grid_size
        P_grid = self._build_P_grid(*grid_size)
        P_hat_grid = torch.tensor(self._build_P_hat(self.num_fiducial, self.C,
                                     P_grid)).float()
        # self.P_hat_grid = P_hat_grid
        self.register_buffer("P_hat_grid", P_hat_grid)
     

    def _build_C_edge(self, num_fiducial):
        n = num_fiducial//2
        ctrl_pts_x = np.linspace(-1.0, 1.0, n)*self.fiducial_width
        ctrl_pts_y_top = -1 * np.ones(n) * self.fiducial_height
        ctrl_pts_y_bottom = np.ones(n) * self.fiducial_height
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        # if not self.head_tail:
        C = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        return C  # num_fiducial x 2



    def _build_C_cross(self):

        ctrl_pts_x = np.linspace(-1.0, 1.0, 3) * self.fiducial_width
        ctrl_pts_y_top = -1 * np.ones(3) * self.fiducial_height
        ctrl_pts_y_bottom = np.ones(3) * self.fiducial_height
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        ctrl_pts_x_center = np.linspace(-1.0,1.0,5)[[1,3]]*self.fiducial_width
        ctrl_pts_y_center = np.zeros(2)
        ctrl_pts_center = np.stack([ctrl_pts_x_center, ctrl_pts_y_center], axis=1)
        C = np.concatenate([ctrl_pts_top,ctrl_pts_center, ctrl_pts_bottom], axis=0)
        return C


    def _build_C_center(self):
        n = 6
        ctrl_pts_x = np.linspace(-1.0, 1.0, n) * self.fiducial_width
        ctrl_pts_y_top = -1 * np.ones(n) * self.fiducial_height
        ctrl_pts_y_bottom = np.ones(n) * self.fiducial_height
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        center_line = (ctrl_pts_top + ctrl_pts_bottom)/2
        center_line = center_line[1:-1]
        C = np.concatenate([ctrl_pts_top[[0,-1]],center_line, ctrl_pts_bottom[[0,-1]]])
        return C  # num_fiducial x 2

    def _build_P_grid(self, h, w):
        fiducial_grid_x = np.linspace(-1, 1, w) * self.fiducial_width
        fiducial_grid_y = np.linspace(-1, 1, h) * self.fiducial_height
        P = np.stack(  # self.fiducial_w x self.fiducial_h x 2
            np.meshgrid(fiducial_grid_x, fiducial_grid_y),
            axis=2)
        return P.reshape([
            -1, 2
        ])

    def _build_inv_delta_C(self, num_fiducial, C):
        """Return inv_delta_C which is needed to calculate T."""
        hat_C = np.zeros((num_fiducial, num_fiducial), dtype=float)
        for i in range(0, num_fiducial):
            for j in range(i, num_fiducial):
                r = np.linalg.norm(C[i] - C[j])
                hat_C[i, j] = r
                hat_C[j, i] = r
        np.fill_diagonal(hat_C, 1)
        hat_C = (hat_C**2) * np.log(hat_C)
        # print(C.shape, hat_C.shape)
        delta_C = np.concatenate(  # num_fiducial+3 x num_fiducial+3
            [
                np.concatenate([np.ones((num_fiducial, 1)), C, hat_C],
                               axis=1),  # num_fiducial x num_fiducial+3
                np.concatenate([np.zeros(
                    (2, 3)), np.transpose(C)], axis=1),  # 2 x num_fiducial+3
                np.concatenate([np.zeros(
                    (1, 3)), np.ones((1, num_fiducial))],
                               axis=1)  # 1 x num_fiducial+3
            ],
            axis=0)
        inv_delta_C = np.linalg.inv(delta_C)
        return inv_delta_C  # num_fiducial+3 x num_fiducial+3

    def _build_P(self, num_pts):
        fiducial_grid_x = np.linspace(-1.0, 1.0, int(num_pts / 2))*self.fiducial_width
        # fiducial_grid_y = (
        #     np.arange(-fiducial_height, fiducial_height, 2) +
        #     1.0) / fiducial_height  # self.fiducial_height
        # P = np.stack(  # self.fiducial_w x self.fiducial_h x 2
        #     np.meshgrid(fiducial_grid_x, fiducial_grid_y),
        #     axis=2)
        ctrl_pts_y_top = -1 * np.ones(fiducial_grid_x.shape[0])*self.fiducial_height
        ctrl_pts_y_bottom = np.ones(fiducial_grid_x.shape[0])*self.fiducial_height
        ctrl_pts_top = np.stack([fiducial_grid_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([fiducial_grid_x, ctrl_pts_y_bottom], axis=1)
        P = np.concatenate([ctrl_pts_top, ctrl_pts_bottom[::-1]], axis=0)
        return P.reshape([
            -1, 2
        ])  # n (= self.fiducial_width x self.fiducial_height) x 2

    def _build_P_hat(self, num_fiducial, C, P):
        n = P.shape[
            0]  # n (= self.fiducial_width x self.fiducial_height)
        P_tile = np.tile(np.expand_dims(P, axis=1),
                         (1, num_fiducial,
                          1))  # n x 2 -> n x 1 x 2 -> n x num_fiducial x 2
        C_tile = np.expand_dims(C, axis=0)  # 1 x num_fiducial x 2
        P_diff = P_tile - C_tile  # n x num_fiducial x 2
        rbf_norm = np.linalg.norm(
            P_diff, ord=2, axis=2, keepdims=False)  # n x num_fiducial
        rbf = np.multiply(np.square(rbf_norm),
                          np.log(rbf_norm + self.eps))  # n x num_fiducial
        P_hat = np.concatenate([np.ones((n, 1)), P, rbf], axis=1)
        return P_hat  # n x num_fiducial+3


    def build_inv_P(self,num_fiducial, P, C):
        p_hat = self._build_P_hat(num_fiducial, C, P) # n x (num_fiducial +3)
        p_hat = np.concatenate([p_hat,
                                np.concatenate([np.zeros(
                                    (2, 3)), np.transpose(C)], axis=1),  # 2 x num_fiducial+3
                                np.concatenate([np.zeros(
                                    (1, 3)), np.ones((1, num_fiducial))],
                                    axis=1)  # 1 x num_fiducial+3
                                ])
        inv_p_hat = np.linalg.pinv(p_hat) #(num_fiducial +3) x (n +3)
        return inv_p_hat


    def solve_T(self, batch_C_prime, batch_P=None):
        device = self.inv_delta_C.device
        if batch_P is None: # solve with control point pair
            batch_size = batch_C_prime.shape[0]
            batch_inv_delta_C = self.inv_delta_C.repeat(batch_size, 1, 1)

        else: # solve with least square method
            batch_size = batch_C_prime.size(0)
            batch_inv_delta_C = torch.from_numpy(self.build_inv_P(self.num_fiducial, batch_P, self.C))[None].repeat(batch_size, 1,
                                                                                                  1).float()
        if not isinstance(batch_C_prime, torch.Tensor):
            batch_C_prime = torch.from_numpy(batch_C_prime)
        batch_C_prime_with_zeros = torch.cat(
            (batch_C_prime, torch.zeros(batch_size, 3, 2).float().to(device)),
            dim=1)  # batch_size x num_fiducial+3 x 2
        batch_T = torch.bmm(
            batch_inv_delta_C,
            batch_C_prime_with_zeros)  # batch_size x num_fiducial+3 x 2
        return batch_T


    def build_P_border(self, batch_T):
        batch_T = batch_T.view(-1, self.num_fiducial+3, 2)
        batch_P_hat = self.P_hat.repeat(batch_T.shape[0], 1, 1)
        batch_P_boder = torch.bmm(batch_P_hat, batch_T) # batch_size x n x 2
        return batch_P_boder

    def inverse_P_border(self, batch_P_boder):
        batch_P_hat_inv = self.P_hat_inv.repeat(batch_P_boder.shape[0], 1, 1)
        batch_T = torch.bmm(batch_P_hat_inv, batch_P_boder)  # 逆变换
        batch_T = batch_T.view(-1, self.num_fiducial + 3, 2)
        return batch_T
    
    def build_P_grid(self, batch_T):
        batch_T = batch_T.view(-1, self.num_fiducial + 3, 2)
        batch_P_hat_grid = self.P_hat_grid.repeat(batch_T.shape[0], 1, 1)
        batch_P_grid = torch.bmm(batch_P_hat_grid, batch_T)
        return batch_P_grid # batch_size x n x 2


    # def build_P_prime(self, batch_C_prime, device='cpu'):
    #     """Generate Grid from batch_C_prime [batch_size x num_fiducial x 2]"""
    #     batch_size = batch_C_prime.size(0)
    #     batch_inv_delta_C = self.inv_delta_C.repeat(batch_size, 1, 1)
    #     batch_P_hat = self.P_hat.repeat(batch_size, 1, 1)
    #     batch_P_hat_grid = self.P_hat_grid.repeat(batch_size,1,1)
    #     batch_C_prime_with_zeros = torch.cat(
    #         (batch_C_prime, torch.zeros(batch_size, 3, 2).float().to(device)),
    #         dim=1)  # batch_size x num_fiducial+3 x 2
    #     batch_T = torch.bmm(
    #         batch_inv_delta_C,
    #         batch_C_prime_with_zeros)  # batch_size x num_fiducial+3 x 2
    #     batch_P_boder = torch.bmm(batch_P_hat, batch_T)  # batch_size x n x 2
    #     batch_P_grid = torch.bmm(batch_P_hat_grid, batch_T)
    #     return batch_T, batch_P_boder, batch_P_grid  # batch_size x n x 2
    #
    # def build_P_prime_p(self, batch_C_prime, P, device='cpu'):
    #     batch_size = batch_C_prime.size(0)
    #     # batch_inv_delta_C = self.inv_delta_C[direction].repeat(batch_size, 1, 1)
    #     inv_p = torch.from_numpy(self.build_inv_P(self.num_fiducial, P, self.C))[None].repeat(batch_size, 1, 1).float()
    #     batch_P_hat = self.P_hat.repeat(batch_size, 1, 1)
    #     batch_P_hat_grid = self.P_hat_grid.repeat(batch_size, 1, 1)
    #     batch_C_prime_with_zeros = torch.cat(
    #         (batch_C_prime, torch.zeros(batch_size, 3, 2).float().to(device)),
    #         dim=1)  # batch_size x num_fiducial+3 x 2
    #     batch_T = torch.bmm(
    #         inv_p,
    #         batch_C_prime_with_zeros)  # batch_size x num_fiducial+3 x 2
    #     batch_P_boder = torch.bmm(batch_P_hat, batch_T)  # batch_size x n x 2
    #     batch_P_grid = torch.bmm(batch_P_hat_grid, batch_T)
    #     pc = self._build_P_hat(self.num_fiducial, self.C, self.C)
    #     pc = torch.from_numpy(pc)
    #     batch_pc = pc.repeat(batch_size, 1, 1)
    #     control_pts = torch.bmm(batch_pc.float(), batch_T)
    #     # batch_T, batch_P_boder, batch_P_grid = self.build_P_prime(control_pts)
    #     sp_hat = torch.from_numpy(self._build_P_hat(self.num_fiducial, self.C, P)).repeat(batch_size, 1, 1)
    #     fit_p = torch.bmm(sp_hat.float(), batch_T)
    #     res = batch_C_prime - fit_p
    #     res = np.abs(res).sum()
    #     return batch_T, batch_P_boder, batch_P_grid, control_pts  # batch_size x n x 2


