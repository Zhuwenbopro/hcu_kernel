#include <ATen/ATen.h>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>
#include "../reduce.h"

#include <cmath>
#include <cstdint>
#include <type_traits>

#ifdef TORCH_VERSION_OVER_290
#define C10_WARP_SIZE C10_WARP_SIZE_STATIC
#endif

namespace at {
namespace native {

template <typename scalar_t>
__device__ __forceinline__ float gated_to_float(const scalar_t value) {
  return static_cast<float>(value);
}

template <typename scalar_t>
__device__ __forceinline__ void gated_from_float(scalar_t& out, const float value) {
  out = static_cast<scalar_t>(value);
}

__device__ __forceinline__ float gated_sigmoid(const float value) {
#ifdef USE_ROCM
  constexpr float log2e = 1.44269504088896340736f;
  return 1.0f / (1.0f + __builtin_amdgcn_exp2f(-value * log2e));
#else
  return 1.0f / (1.0f + expf(-value));
#endif
}

template <typename T, int THREADS_PER_ROW>
__device__ __forceinline__ T gated_row_reduce_sum(T value) {
#pragma unroll
  for (int mask = THREADS_PER_ROW / 2; mask > 0; mask >>= 1) {
#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
    value += __shfl_xor(value, mask, THREADS_PER_ROW);
#else
    value += __shfl_xor_sync(0xffffffff, value, mask, THREADS_PER_ROW);
#endif
  }
  return value;
}

template <typename T, int THREADS_PER_ROW>
__device__ __forceinline__ T gated_row_reduce_max(T value) {
#pragma unroll
  for (int mask = THREADS_PER_ROW / 2; mask > 0; mask >>= 1) {
#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
    value = fmaxf(value, __shfl_xor(value, mask, THREADS_PER_ROW));
#else
    value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, mask, THREADS_PER_ROW));
#endif
  }
  return value;
}

template <typename scalar_t, int VEC, int THREADS_PER_ROW, int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE) void fuse_rms_norm_gated_per_token_quant_int8_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ z,
    int8_t* __restrict__ output,
    float* __restrict__ scales,
    int64_t stride_x,
    int64_t stride_z,
    int64_t rows,
    int64_t cols,
    float eps) {
  constexpr int vec_cols = 128 / VEC;
  __shared__ float max_shared[BLOCK_SIZE / C10_WARP_SIZE];
  __shared__ float token_absmax;

  const int tid = threadIdx.x;
  const int lane = tid % THREADS_PER_ROW;
  const int head = tid / THREADS_PER_ROW;
  const int heads = cols / 128;
  const int64_t row = blockIdx.x;
  const int64_t col = static_cast<int64_t>(head) * 128 + lane * VEC;

  float x_cache[VEC];
  float z_cache[VEC];
  float local_sq = 0.0f;
  if (head < heads && lane < vec_cols) {
    scalar_t x_vec[VEC];
    scalar_t z_vec[VEC];
    using LoadT = at::native::memory::aligned_vector<scalar_t, VEC>;
    *(LoadT*)x_vec = *(const LoadT*)(x + row * stride_x + col);
    *(LoadT*)z_vec = *(const LoadT*)(z + row * stride_z + col);
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
      x_cache[i] = gated_to_float(x_vec[i]);
      z_cache[i] = gated_to_float(z_vec[i]);
      local_sq += x_cache[i] * x_cache[i];
    }
  }

  local_sq = gated_row_reduce_sum<float, THREADS_PER_ROW>(local_sq);
  // Preserve the original head-local sum and normalization order, then
  // reduce absmax over every head belonging to the same projection token.
  const float rstd = rsqrtf(local_sq / 128.0f + eps);
  float y_cache[VEC];
  float local_absmax = 0.0f;
  if (head < heads && lane < vec_cols) {
    scalar_t w_vec[VEC];
    using LoadT = at::native::memory::aligned_vector<scalar_t, VEC>;
    *(LoadT*)w_vec = *(const LoadT*)(weight + lane * VEC);
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
      const float normed = x_cache[i] * rstd * gated_to_float(w_vec[i]);
      const float gated = normed * (z_cache[i] * gated_sigmoid(z_cache[i]));
      scalar_t rounded;
      gated_from_float(rounded, gated);
      y_cache[i] = gated_to_float(rounded);
      local_absmax = fmaxf(local_absmax, fabsf(y_cache[i]));
    }
  }

  local_absmax = BlockReduceMax_ROW<float, BLOCK_SIZE>(local_absmax, max_shared);
  if (tid == 0) {
    token_absmax = fmaxf(local_absmax, 1.0e-10f);
    scales[row] = token_absmax / 127.0f;
  }
  __syncthreads();

  if (head < heads && lane < vec_cols) {
    const float inv_scale = 127.0f / token_absmax;
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
      output[row * cols + col + i] = static_cast<int8_t>(
          nearbyintf(y_cache[i] * inv_scale));
    }
  }
}

void fuse_rms_norm_gated_per_token_quant_int8(
    torch::Tensor& output,
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& z,
    torch::Tensor& scales,
    double eps) {
  TORCH_CHECK(x.is_cuda() && weight.is_cuda() && z.is_cuda() && output.is_cuda() &&
                  scales.is_cuda(),
              "all tensors must be CUDA/HIP tensors");
  TORCH_CHECK(x.dim() == 2 && z.sizes() == x.sizes(),
              "x and z must be [tokens, heads * 128] with matching shapes");
  TORCH_CHECK(x.size(1) > 0 && x.size(1) % 128 == 0 && x.size(1) <= 8192,
              "GDN fused small-head kernel requires 1..64 heads of dim 128");
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == 128,
              "weight must have shape [128]");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kHalf,
              "x must be BF16 or FP16");
  TORCH_CHECK(z.scalar_type() == x.scalar_type() && weight.scalar_type() == x.scalar_type(),
              "x, weight and z must have the same dtype");
  TORCH_CHECK(x.is_contiguous() && z.is_contiguous() && weight.is_contiguous(),
              "x, weight and z must be contiguous");
  TORCH_CHECK(x.device() == weight.device() && x.device() == z.device() &&
                  x.device() == output.device() && x.device() == scales.device(),
              "all tensors must be on the same device");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(z.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(weight.data_ptr()) % 16 == 0,
              "x, z and weight pointers must be aligned to 16 bytes");
  TORCH_CHECK(output.scalar_type() == at::kChar && output.sizes() == x.sizes() &&
                  output.is_contiguous(),
              "output must be contiguous int8 with x's shape");
  TORCH_CHECK(scales.scalar_type() == at::kFloat && scales.dim() == 2 &&
                  scales.size(0) == x.size(0) && scales.size(1) == 1 &&
                  scales.is_contiguous(),
              "scales must be contiguous float32 [rows, 1]");

  const auto device_guard = at::cuda::OptionalCUDAGuard(device_of(x));
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t rows = x.size(0);
  if (rows == 0) return;
  constexpr int threads_per_row = 16;
#define GDN_LAUNCH(BLOCK) \
  fuse_rms_norm_gated_per_token_quant_int8_kernel<scalar_t, 8, threads_per_row, BLOCK> \
      <<<rows, BLOCK, 0, stream>>>( \
          x.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(), \
          z.data_ptr<scalar_t>(), output.data_ptr<int8_t>(), \
          scales.data_ptr<float>(), x.stride(0), z.stride(0), rows, \
          x.size(1), static_cast<float>(eps))
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(),
      "fuse_rms_norm_gated_per_token_quant_int8", [&] {
        const auto threads = (x.size(1) / 128) * threads_per_row;
        if (threads <= 64) { GDN_LAUNCH(64); }
        else if (threads <= 128) { GDN_LAUNCH(128); }
        else if (threads <= 256) { GDN_LAUNCH(256); }
        else if (threads <= 512) { GDN_LAUNCH(512); }
        else { GDN_LAUNCH(1024); }
      });
#undef GDN_LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace native
}  // namespace at
