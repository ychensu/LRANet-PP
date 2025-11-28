#pragma once
#include <torch/types.h>
#include <torch/extension.h>
#include "GridSamplerBatch.h"
//#include "cuda/GridSamplerBatch.cuh"


//namespace at{ namespace native{
Tensor grid_sampler_batch_2d(const at::Tensor& input, const at::Tensor& grid, const at::Tensor& batch_idx,
                           int64_t interpolation_mode, int64_t padding_mode,
                           bool align_corners){
//                           return grid_sampler_batch_2d_cpu(input, grid, batch_idx, interpolation_mode, padding_mode, align_corners);
                          if (input.type().is_cuda()) {
//                            #ifdef WITH_CUDA
                                return grid_sampler_batch_2d_cuda(input, grid, batch_idx, interpolation_mode, padding_mode, align_corners);
//                            #else
//                                AT_ERROR("Not compiled with GPU support");
//                            #endif
                          }
                          return grid_sampler_batch_2d_cpu(input, grid, batch_idx, interpolation_mode, padding_mode, align_corners);
                        }

std::tuple<at::Tensor, at::Tensor>
grid_sampler_batch_2d_backward(const at::Tensor& grad_output, const at::Tensor& input, const at::Tensor& grid, const at::Tensor& batch_idx,
                             int64_t interpolation_mode, int64_t padding_mode, bool align_corners){
                             if(grad_output.type().is_cuda()){
//                             #ifdef WITH_CUDA
                               return grid_sampler_batch_2d_backward_cuda(grad_output, input, grid, batch_idx, interpolation_mode, padding_mode, align_corners);
//                             #else
//                                AT_ERROR("Not compiled with GPU support");
//                             #endif
                             }
                             return grid_sampler_batch_2d_backward_cpu(grad_output, input, grid, batch_idx,
                             interpolation_mode, padding_mode, align_corners);
                             }


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("grid_sampler_batch_2d", & grid_sampler_batch_2d, "grid_sampler_2d_cpu");
  m.def("grid_sampler_batch_2d_backward", & grid_sampler_batch_2d_backward, "grid_sampler_2d_backward_cpu");
}
//}
//}