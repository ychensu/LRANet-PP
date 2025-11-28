#include <torch/extension.h>
#include <ATen/TensorUtils.h>
#include <ATen/ATen.h>
#include <ATen/Device.h>
#include <ATen/NativeFunctions.h>
//#include <ATen/Parallel.h>
#include <c10/core/Layout.h>
//#include <ATen/cpu/vml.h>
#include <ATen/native/IndexingUtils.h>
#include <ATen/native/UpSample.h>
//#include <ATen/native/cpu/GridSamplerKernel.h>
#include <c10/util/Exception.h>
#include "GridSamplerBatch.h"

//namespace at {namespace native{
using namespace at::native;
using namespace at;
using dd::GridSamplerInterpolation;
using dd::GridSamplerPadding;


at::Tensor _grid_sampler_batch_2d_cpu_fallback(const at::Tensor& input, const at::Tensor& grid, const at::Tensor& batch_idx,
                                     int64_t interpolation_mode_,
                                     int64_t padding_mode_,
                                     bool align_corners) {
  auto interpolation_mode = static_cast<GridSamplerInterpolation>(interpolation_mode_);
  auto padding_mode = static_cast<GridSamplerPadding>(padding_mode_);
  using scalar_t = float;

  int64_t N = input.size(0);
  int64_t C = input.size(1);
  int64_t inp_H = input.size(2);
  int64_t inp_W = input.size(3);
  int64_t out_H = grid.size(1);
  int64_t out_W = grid.size(2);
//  auto output = at::empty({N, C, out_H, out_W}, input.options());

  int64_t out_N = batch_idx.size(0);
  auto output = at::empty({out_N, C, out_H, out_W}, input.options());

  int64_t inp_sN = input.stride(0);
  int64_t inp_sC = input.stride(1);
  int64_t inp_sH = input.stride(2);
  int64_t inp_sW = input.stride(3);
  int64_t grid_sN = grid.stride(0);
  int64_t grid_sH = grid.stride(1);
  int64_t grid_sW = grid.stride(2);
  int64_t grid_sCoor = grid.stride(3);
  int64_t out_sN = output.stride(0);
  int64_t out_sC = output.stride(1);
  int64_t out_sH = output.stride(2);
  int64_t out_sW = output.stride(3);
  scalar_t *inp_ptr = input.data_ptr<scalar_t>();
  scalar_t *out_ptr = output.data_ptr<scalar_t>();
  scalar_t *grid_ptr = grid.data_ptr<scalar_t>();

  scalar_t *batch_idx_ptr = batch_idx.data_ptr<scalar_t>();
  // loop over each output pixel
  at::parallel_for(0, out_N, 0, [&](int64_t start, int64_t end) {
    for (int64_t n = start; n < end; ++n) {
      scalar_t *grid_ptr_N = grid_ptr + n * grid_sN;
      int64_t batch_idx_N = batch_idx_ptr[n];
      //scalar_t *inp_ptr_N = inp_ptr + n * inp_sN;
      scalar_t *inp_ptr_N = inp_ptr + batch_idx_N * inp_sN;
      for (int64_t h = 0; h < out_H; ++h) {
        for (int64_t w = 0; w < out_W; ++w) {
          // get the corresponding input x, y, z co-ordinates from grid
          scalar_t *grid_ptr_NHW = grid_ptr_N + h * grid_sH + w * grid_sW;
          scalar_t x = *grid_ptr_NHW;
          scalar_t y = grid_ptr_NHW[grid_sCoor];

          scalar_t ix = grid_sampler_compute_source_index(x, inp_W, padding_mode, align_corners);
          scalar_t iy = grid_sampler_compute_source_index(y, inp_H, padding_mode, align_corners);

          if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
            // get corner pixel values from (x, y)
            // for 4d, we use north-east-south-west
            int64_t ix_nw = static_cast<int64_t>(std::floor(ix));
            int64_t iy_nw = static_cast<int64_t>(std::floor(iy));

            int64_t ix_ne = ix_nw + 1;
            int64_t iy_ne = iy_nw;

            int64_t ix_sw = ix_nw;
            int64_t iy_sw = iy_nw + 1;

            int64_t ix_se = ix_nw + 1;
            int64_t iy_se = iy_nw + 1;


            // get surfaces to each neighbor:
            scalar_t nw = (ix_se - ix)    * (iy_se - iy);
            scalar_t ne = (ix    - ix_sw) * (iy_sw - iy);
            scalar_t sw = (ix_ne - ix)    * (iy    - iy_ne);
            scalar_t se = (ix    - ix_nw) * (iy    - iy_nw);

            // calculate bilinear weighted pixel value and set output pixel
            scalar_t *inp_ptr_NC = inp_ptr_N;
            scalar_t *out_ptr_NCHW = out_ptr + n * out_sN + h * out_sH + w * out_sW;
            for (int64_t c = 0; c < C; ++c, out_ptr_NCHW += out_sC, inp_ptr_NC += inp_sC) {
              auto res = static_cast<scalar_t>(0);
              if (within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W)) {
                res += inp_ptr_NC[iy_nw * inp_sH + ix_nw * inp_sW] * nw;
              }
              if (within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W)) {
                res += inp_ptr_NC[iy_ne * inp_sH + ix_ne * inp_sW] * ne;
              }
              if (within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W)) {
                res += inp_ptr_NC[iy_sw * inp_sH + ix_sw * inp_sW] * sw;
              }
              if (within_bounds_2d(iy_se, ix_se, inp_H, inp_W)) {
                res += inp_ptr_NC[iy_se * inp_sH + ix_se * inp_sW] * se;
              }
              *out_ptr_NCHW = res;
            }
          } else if (interpolation_mode == GridSamplerInterpolation::Nearest) {
            int64_t ix_nearest = static_cast<int64_t>(std::nearbyint(ix));
            int64_t iy_nearest = static_cast<int64_t>(std::nearbyint(iy));

            // assign nearest neighor pixel value to output pixel
            scalar_t *out_ptr_NCHW = out_ptr + n * out_sN + h * out_sH + w * out_sW;
            scalar_t *inp_ptr_NC = inp_ptr_N;
            for (int64_t c = 0; c < C; ++c, out_ptr_NCHW += out_sC, inp_ptr_NC += inp_sC) {
              if (within_bounds_2d(iy_nearest, ix_nearest, inp_H, inp_W)) {
                *out_ptr_NCHW = inp_ptr_NC[iy_nearest * inp_sH + ix_nearest * inp_sW];
              } else {
                *out_ptr_NCHW = static_cast<scalar_t>(0);
              }
            }
          } else if (interpolation_mode == GridSamplerInterpolation::Bicubic) {
            // grid_sampler_compute_source_index will "clip the value" of idx depends on the padding,
            // which would cause calculation to be wrong,
            // for example x = -0.1 -> ix = 0 for zero padding, but in bicubic ix = floor(x) = -1
            // There would be more problem in reflection padding, since the -1 and +1 direction is not fixed in boundary condition
            ix = grid_sampler_unnormalize(x, inp_W, align_corners);
            iy = grid_sampler_unnormalize(y, inp_H, align_corners);

            scalar_t ix_nw = std::floor(ix);
            scalar_t iy_nw = std::floor(iy);

            const scalar_t tx = ix - ix_nw;
            const scalar_t ty = iy - iy_nw;

            scalar_t *inp_ptr_NC = inp_ptr_N;
            scalar_t *out_ptr_NCHW = out_ptr + n * out_sN + h * out_sH + w * out_sW;
            for (int64_t c = 0; c < C; ++c, out_ptr_NCHW += out_sC, inp_ptr_NC += inp_sC) {
              // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
              scalar_t coefficients[4];

              // Interpolate 4 values in the x directon
              for (int64_t i = 0; i < 4; ++i) {
                coefficients[i] = at::native::cubic_interp1d<scalar_t>(
                  get_value_bounded<scalar_t>(inp_ptr_NC, ix_nw - 1, iy_nw - 1 + i, inp_W, inp_H, inp_sW, inp_sH, padding_mode, align_corners),
                  get_value_bounded<scalar_t>(inp_ptr_NC, ix_nw + 0, iy_nw - 1 + i, inp_W, inp_H, inp_sW, inp_sH, padding_mode, align_corners),
                  get_value_bounded<scalar_t>(inp_ptr_NC, ix_nw + 1, iy_nw - 1 + i, inp_W, inp_H, inp_sW, inp_sH, padding_mode, align_corners),
                  get_value_bounded<scalar_t>(inp_ptr_NC, ix_nw + 2, iy_nw - 1 + i, inp_W, inp_H, inp_sW, inp_sH, padding_mode, align_corners),
                  tx);
              }

              // Interpolate in the y direction
              *out_ptr_NCHW = at::native::cubic_interp1d<scalar_t>(
                coefficients[0],
                coefficients[1],
                coefficients[2],
                coefficients[3],
                ty);
            }
          }
        }
      }
    }
  });
  return output;
}

std::tuple<at::Tensor, at::Tensor>
_grid_sampler_batch_2d_cpu_fallback_backward(const at::Tensor& grad_output,
                                       const at::Tensor& input, const at::Tensor& grid, const at::Tensor& batch_idx,
                                       int64_t interpolation_mode_,
                                       int64_t padding_mode_,
                                       bool align_corners) {
  const auto interpolation_mode = static_cast<GridSamplerInterpolation>(interpolation_mode_);
  const auto padding_mode = static_cast<GridSamplerPadding>(padding_mode_);
  using scalar_t = float;

  auto grad_input = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto grad_grid = at::empty_like(grid, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  // If interpolation mode is Nearest, then grad_grid is not filled in the
  // loop below.
  if (interpolation_mode == GridSamplerInterpolation::Nearest) {
    grad_grid.zero_();
  }
  int64_t N = input.size(0);
  int64_t C = input.size(1);
  int64_t inp_H = input.size(2);
  int64_t inp_W = input.size(3);
  int64_t out_H = grid.size(1);
  int64_t out_W = grid.size(2);
  int64_t inp_sN = input.stride(0);
  int64_t inp_sC = input.stride(1);
  int64_t inp_sH = input.stride(2);
  int64_t inp_sW = input.stride(3);
  int64_t grid_sN = grid.stride(0);
  int64_t grid_sH = grid.stride(1);
  int64_t grid_sW = grid.stride(2);
  int64_t grid_sCoor = grid.stride(3);
  int64_t gOut_sN = grad_output.stride(0);
  int64_t gOut_sC = grad_output.stride(1);
  int64_t gOut_sH = grad_output.stride(2);
  int64_t gOut_sW = grad_output.stride(3);
  int64_t gInp_sN = grad_input.stride(0);
  int64_t gInp_sC = grad_input.stride(1);
  int64_t gInp_sH = grad_input.stride(2);
  int64_t gInp_sW = grad_input.stride(3);
  int64_t gGrid_sN = grad_grid.stride(0);
  int64_t gGrid_sW = grad_grid.stride(2);
  scalar_t *inp_ptr = input.data_ptr<scalar_t>();
  scalar_t *grid_ptr = grid.data_ptr<scalar_t>();
  scalar_t *gOut_ptr = grad_output.data_ptr<scalar_t>();
  scalar_t *gInp_ptr = grad_input.data_ptr<scalar_t>();
  scalar_t *gGrid_ptr = grad_grid.data_ptr<scalar_t>();

  int64_t out_N = grid.size(0);
  scalar_t *gBatch_idx_ptr = batch_idx.data_ptr<scalar_t>();
  // loop over each output pixel
//  at::parallel_for(0, N, 0, [&](int64_t start, int64_t end) {
  at::parallel_for(0, out_N, 0, [&](int64_t start, int64_t end) {
    for (int64_t n = start; n < end; ++n) {
      scalar_t *grid_ptr_N = grid_ptr + n * grid_sN;
//      scalar_t *inp_ptr_N = inp_ptr + n * inp_sN;

      int64_t n_batch_idx = gBatch_idx_ptr[n];
      scalar_t *inp_ptr_N = inp_ptr + n_batch_idx * inp_sN;
      scalar_t *gGrid_ptr_NHW = gGrid_ptr + n * gGrid_sN;
      for (int64_t h = 0; h < out_H; ++h) {
        for (int64_t w = 0; w < out_W; ++w, gGrid_ptr_NHW += gGrid_sW /* grad_grid is contiguous */ ) {
          // get the corresponding input x, y co-ordinates from grid
          scalar_t *grid_ptr_NHW = grid_ptr_N + h * grid_sH + w * grid_sW;
          scalar_t x = *grid_ptr_NHW;
          scalar_t y = grid_ptr_NHW[grid_sCoor];

          // multipliers for gradients on ix, iy
          // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
          scalar_t gix_mult, giy_mult;
          scalar_t ix = grid_sampler_compute_source_index_set_grad(x, inp_W, padding_mode, align_corners, &gix_mult);
          scalar_t iy = grid_sampler_compute_source_index_set_grad(y, inp_H, padding_mode, align_corners, &giy_mult);

          if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
            // get corner pixel values from (x, y)
            // for 4d, we use north-east-south-west
            int64_t ix_nw = static_cast<int64_t>(std::floor(ix));
            int64_t iy_nw = static_cast<int64_t>(std::floor(iy));

            int64_t ix_ne = ix_nw + 1;
            int64_t iy_ne = iy_nw;

            int64_t ix_sw = ix_nw;
            int64_t iy_sw = iy_nw + 1;

            int64_t ix_se = ix_nw + 1;
            int64_t iy_se = iy_nw + 1;

            // get surfaces to each neighbor:
            scalar_t nw = (ix_se - ix)    * (iy_se - iy);
            scalar_t ne = (ix    - ix_sw) * (iy_sw - iy);
            scalar_t sw = (ix_ne - ix)    * (iy    - iy_ne);
            scalar_t se = (ix    - ix_nw) * (iy    - iy_nw);

            scalar_t gix = static_cast<scalar_t>(0), giy = static_cast<scalar_t>(0);
            scalar_t *gOut_ptr_NCHW = gOut_ptr + n * gOut_sN + h * gOut_sH + w * gOut_sW;
//            scalar_t *gInp_ptr_NC = gInp_ptr + n * gInp_sN;
            scalar_t *gInp_ptr_NC = gInp_ptr + n_batch_idx * gInp_sN;
            scalar_t *inp_ptr_NC = inp_ptr_N;
            // calculate bilinear weighted pixel value and set output pixel
            for (int64_t c = 0; c < C; ++c, gOut_ptr_NCHW += gOut_sC, gInp_ptr_NC += gInp_sC, inp_ptr_NC += inp_sC) {
              scalar_t gOut = *gOut_ptr_NCHW;

              // calculate and set grad_input
              safe_add_2d(gInp_ptr_NC, iy_nw, ix_nw, gInp_sH, gInp_sW, inp_H, inp_W, nw * gOut);
              safe_add_2d(gInp_ptr_NC, iy_ne, ix_ne, gInp_sH, gInp_sW, inp_H, inp_W, ne * gOut);
              safe_add_2d(gInp_ptr_NC, iy_sw, ix_sw, gInp_sH, gInp_sW, inp_H, inp_W, sw * gOut);
              safe_add_2d(gInp_ptr_NC, iy_se, ix_se, gInp_sH, gInp_sW, inp_H, inp_W, se * gOut);

              // calculate grad_grid
              if (within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W)) {
                scalar_t nw_val = inp_ptr_NC[iy_nw * inp_sH + ix_nw * inp_sW];
                gix -= nw_val * (iy_se - iy) * gOut;
                giy -= nw_val * (ix_se - ix) * gOut;
              }
              if (within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W)) {
                scalar_t ne_val = inp_ptr_NC[iy_ne * inp_sH + ix_ne * inp_sW];
                gix += ne_val * (iy_sw - iy) * gOut;
                giy -= ne_val * (ix - ix_sw) * gOut;
              }
              if (within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W)) {
                scalar_t sw_val = inp_ptr_NC[iy_sw * inp_sH + ix_sw * inp_sW];
                gix -= sw_val * (iy - iy_ne) * gOut;
                giy += sw_val * (ix_ne - ix) * gOut;
              }
              if (within_bounds_2d(iy_se, ix_se, inp_H, inp_W)) {
                scalar_t se_val = inp_ptr_NC[iy_se * inp_sH + ix_se * inp_sW];
                gix += se_val * (iy - iy_nw) * gOut;
                giy += se_val * (ix - ix_nw) * gOut;
              }
            }

            // assuming grad_grid is contiguous
            gGrid_ptr_NHW[0] = gix_mult * gix;
            gGrid_ptr_NHW[1] = giy_mult * giy;
          } else if (interpolation_mode == GridSamplerInterpolation::Nearest) {
            int64_t ix_nearest = static_cast<int64_t>(std::nearbyint(ix));
            int64_t iy_nearest = static_cast<int64_t>(std::nearbyint(iy));

            // assign nearest neighor pixel value to output pixel
            scalar_t *gOut_ptr_NCHW = gOut_ptr + n * gOut_sN + h * gOut_sH + w * gOut_sW;
//            scalar_t *gInp_ptr_NC = gInp_ptr + n * gInp_sN;
            scalar_t *gInp_ptr_NC = gInp_ptr + n_batch_idx * gInp_sN;
            for (int64_t c = 0; c < C; ++c, gOut_ptr_NCHW += gOut_sC, gInp_ptr_NC += gInp_sC) {
              // calculate and set grad_input
              safe_add_2d(gInp_ptr_NC, iy_nearest, ix_nearest, gInp_sH, gInp_sW,
                          inp_H, inp_W, *gOut_ptr_NCHW);
            }
          } else if (interpolation_mode == GridSamplerInterpolation::Bicubic) {

            ix = grid_sampler_unnormalize_set_grad(x, inp_W, align_corners, &gix_mult);
            iy = grid_sampler_unnormalize_set_grad(y, inp_H, align_corners, &giy_mult);

            scalar_t ix_nw = std::floor(ix);
            scalar_t iy_nw = std::floor(iy);

            const scalar_t tx = ix - ix_nw;
            const scalar_t ty = iy - iy_nw;

            // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
            scalar_t x_coeffs[4];
            // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
            scalar_t y_coeffs[4];
            // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
            scalar_t x_coeffs_grad[4];
            // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
            scalar_t y_coeffs_grad[4];

            get_cubic_upsample_coefficients<scalar_t>(x_coeffs, tx);
            get_cubic_upsample_coefficients<scalar_t>(y_coeffs, ty);
            get_cubic_coefficients_grad<scalar_t>(x_coeffs_grad, tx);
            get_cubic_coefficients_grad<scalar_t>(y_coeffs_grad, ty);

            scalar_t gix = static_cast<scalar_t>(0);
            scalar_t giy = static_cast<scalar_t>(0);

            scalar_t *gOut_ptr_NCHW = gOut_ptr + n * gOut_sN + h * gOut_sH + w * gOut_sW;
//            scalar_t *gInp_ptr_NC = gInp_ptr + n * gInp_sN;
            scalar_t *gInp_ptr_NC = gInp_ptr + n_batch_idx * gInp_sN;
            scalar_t *inp_ptr_NC = inp_ptr_N;

            for (int64_t c = 0; c < C; ++c, gOut_ptr_NCHW += gOut_sC, gInp_ptr_NC += gInp_sC, inp_ptr_NC+= inp_sC) {
              scalar_t gOut = *gOut_ptr_NCHW;

              for (int64_t i = 0; i < 4; ++i) {
                for (int64_t j = 0; j < 4; ++j) {

                  // set input gradient
                  add_value_bounded<scalar_t>(gInp_ptr_NC, ix_nw - 1 + i, iy_nw - 1 + j,
                    inp_W, inp_H, gInp_sW, gInp_sH, gOut * x_coeffs[i] * y_coeffs[j], padding_mode, align_corners);

                  // set grid gradient
                  scalar_t val = get_value_bounded<scalar_t>(inp_ptr_NC, ix_nw - 1 + i, iy_nw - 1 + j,
                    inp_W, inp_H, inp_sW, inp_sH, padding_mode, align_corners);

                  gix -= val * x_coeffs_grad[i] * y_coeffs[j] * gOut;
                  giy -= val * y_coeffs_grad[j] * x_coeffs[i] * gOut;
                }
              }
            }
            gGrid_ptr_NHW[0] = gix_mult * gix;
            gGrid_ptr_NHW[1] = giy_mult * giy;
          }
        }
      }
    }
  });
  return std::make_tuple(grad_input, grad_grid);
}

// No shape checking needed here. See # NOTE [ grid_sampler Native Functions ].
Tensor grid_sampler_batch_2d_cpu(const Tensor& input, const Tensor& grid, const Tensor& batch_idx,
                           int64_t interpolation_mode, int64_t padding_mode,
                           bool align_corners) {

  // AVX gather instructions use signed 32-bit offsets to gather float values.
  // Check for possible overflow and fallback to scalar implementation
  if (input.scalar_type() != kDouble) {
    TORCH_CHECK(input.scalar_type() == kFloat,
                "grid_sampler_2d_cpu not implemented for ", input.scalar_type());
    auto sizes = input.sizes();
    auto strides = input.strides();
    const auto grid_sW = grid.strides()[2];
    // NOTE: Gather offsets are only used for the input H, W dimensions
    //       or only for strided access to the grid Tensor
//    auto max_gather_offset = std::max(
//      (sizes[2] - 1) * strides[2] + (sizes[3] - 1) * strides[3],
//      grid_sW * (at::vml::vec::Vectorized<float>::size() - 1));

//    if (max_gather_offset > std::numeric_limits<int32_t>::max()) {
  return _grid_sampler_batch_2d_cpu_fallback(
    input, grid, batch_idx, interpolation_mode, padding_mode, align_corners);
//    }
  }
//
//  return grid_sampler_2d_cpu_kernel(
//    kCPU, input, grid, interpolation_mode, padding_mode, align_corners);
}

//DEFINE_DISPATCH(grid_sampler_2d_cpu_kernel);



// No shape checking needed here. See # NOTE [ grid_sampler Native Functions ].
std::tuple<Tensor, Tensor>
grid_sampler_batch_2d_backward_cpu(const Tensor& grad_output, const Tensor& input, const Tensor& grid, const Tensor& batch_idx,
                             int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {

  // AVX gather instructions use signed 32-bit offsets to gather float values.
  // Check for possible overflow and fallback to scalar implementation
  if (input.scalar_type() != kDouble) {
    TORCH_CHECK(input.scalar_type() == kFloat,
                "grid_sampler_2d_backward_cpu not implemented for ", input.scalar_type());
    auto isizes = input.sizes();
    auto istrides = input.strides();
    auto gsizes = grad_output.sizes();
    auto gstrides = grad_output.strides();
    const auto grid_sW = grid.strides()[2];
    // NOTE: Gather offsets are only used for the height and width dimensions
//    auto max_gather_offset = std::max(
//      std::max(
//        (isizes[2] - 1) * istrides[2] + (isizes[3] - 1) * istrides[3],
//        (gsizes[2] - 1) * gstrides[2] + (gsizes[3] - 1) * gstrides[3]),
//      grid_sW * (at::vml::vec::Vectorized<float>::size() - 1));

//    if (max_gather_offset > std::numeric_limits<int32_t>::max()) {
  return _grid_sampler_batch_2d_cpu_fallback_backward(
    grad_output, input, grid, batch_idx, interpolation_mode, padding_mode, align_corners);
//    }
  }
//
//  return grid_sampler_2d_backward_cpu_kernel(
//    kCPU, grad_output, input, grid, interpolation_mode, padding_mode, align_corners);
}

//DEFINE_DISPATCH(grid_sampler_2d_backward_cpu_kernel);

// No shape checking needed here. See # NOTE [ grid_sampler Native Functions ].


//PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//  m.def("grid_sampler_2d_cpu", &grid_sampler_2d_cpu, "grid_sampler_2d_cpu");
//  m.def("grid_sampler_2d_backward_cpu", &grid_sampler_2d_backward_cpu, "grid_sampler_2d_backward_cpu");
//}

//}
//}// namespace

//PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//  m.def("grid_sampler_2d_cpu", & grid_sampler_2d_cpu, "grid_sampler_2d_cpu");
//  m.def("grid_sampler_2d_backward_cpu", & grid_sampler_2d_backward_cpu, "grid_sampler_2d_backward_cpu");
//}

//Tensor grid_sampler_batch_2d(const at::Tensor& input, const at::Tensor& grid, const at::Tensor& batch_idx,
//                           int64_t interpolation_mode, int64_t padding_mode,
//                           bool align_corners){
//                           return grid_sampler_batch_2d_cpu(input, grid, batch_idx, interpolation_mode, padding_mode, align_corners);
//                           }
//
//std::tuple<at::Tensor, at::Tensor>
//grid_sampler_batch_2d_backward(const at::Tensor& grad_output, const at::Tensor& input, const at::Tensor& grid, const at::Tensor& batch_idx,
//                             int64_t interpolation_mode, int64_t padding_mode, bool align_corners){
//                             return grid_sampler_batch_2d_backward_cpu(grad_output, input, grid, batch_idx,
//                             interpolation_mode, padding_mode, align_corners);
//                             }