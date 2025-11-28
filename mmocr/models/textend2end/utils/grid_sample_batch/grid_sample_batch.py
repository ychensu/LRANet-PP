import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from . import batch_grid_sample


class _GridSampleBatch(Function):
    @staticmethod
    def forward(ctx, input, grid, batch_idx, interpolation_mode, padding_mode, align_corners):
        ctx.save_for_backward(input,grid, batch_idx)
        ctx.interpolation_mode = interpolation_mode
        ctx.padding_mode = padding_mode
        ctx.align_corners = align_corners
        output = batch_grid_sample.grid_sampler_batch_2d(
            input, grid, batch_idx, interpolation_mode, padding_mode, align_corners
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, grid, batch_idx = ctx.saved_tensors
        interpolation_mode = ctx.interpolation_mode
        padding_mode = ctx.padding_mode
        align_corners = ctx.align_corners
        input_grad, grid_grad = batch_grid_sample.grid_sampler_batch_2d_backward(
            grad_output, input, grid, batch_idx, interpolation_mode, padding_mode, align_corners
        )
        # print(grid_grad.abs().sum())
        grid_grad = grid_grad
        return input_grad, grid_grad, None, None, None, None, None

grid_sample_batch = _GridSampleBatch.apply