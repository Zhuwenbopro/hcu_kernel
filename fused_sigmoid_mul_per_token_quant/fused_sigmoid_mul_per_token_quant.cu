#include <ATen/ATen.h>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <torch/extension.h>

#include <cmath>
#include <cstdint>
#include <limits>

#include "reduce.h"

namespace at {
namespace native {

template <typename scalar_t>
__device__ __forceinline__ float sigmoid_fp32(float value) {
#ifdef USE_ROCM
  constexpr float log2e = 1.44269504088896340736f;
  // Use the native exp2 and reciprocal. Cross-implementation rounding is
  // checked explicitly against the original Triton gating path.
  return __builtin_amdgcn_rcpf(
      1.0f + __builtin_amdgcn_exp2f(-value * log2e));
#else
  return 1.0f / (1.0f + expf(-value));
#endif
}

template <int HEAD_DIM>
__device__ __forceinline__ int64_t sigmoid_gate_offset(
    int64_t offset, int64_t stride_head, int64_t head_dim) {
  if constexpr (HEAD_DIM == 0) {
    return offset;
  } else {
    const int64_t width = HEAD_DIM > 0 ? HEAD_DIM : head_dim;
    const int64_t head = offset / width;
    return head * stride_head + offset - head * width;
  }
}

template <typename scalar_t, int ITEMS, int HEAD_DIM, int BLOCK>
__global__ __launch_bounds__(BLOCK) void fused_sigmoid_mul_per_token_quant_cached_kernel(
    const scalar_t* __restrict__ attn_output,
    const scalar_t* __restrict__ gate,
    int8_t* __restrict__ output,
    float* __restrict__ scales,
    int64_t hidden_dim,
    int64_t gate_stride_row,
    int64_t gate_stride_head,
    int64_t head_dim) {
  const int64_t row = blockIdx.x;
  const int64_t tid = threadIdx.x;
  __shared__ float max_shared[BLOCK / C10_WARP_SIZE];
  __shared__ float token_absmax;
  float values[ITEMS];
  float local_absmax = 0.0f;
#pragma unroll
  for (int i = 0; i < ITEMS; ++i) {
    const int64_t offset = tid + i * BLOCK;
    float value = 0.0f;
    if (offset < hidden_dim) {
      const float x = static_cast<float>(attn_output[row * hidden_dim + offset]);
      const float g = static_cast<float>(gate[
          row * gate_stride_row +
          sigmoid_gate_offset<HEAD_DIM>(offset, gate_stride_head, head_dim)]);
      const scalar_t rounded = static_cast<scalar_t>(x * sigmoid_fp32<scalar_t>(g));
      value = static_cast<float>(rounded);
      local_absmax = fmaxf(local_absmax, fabsf(value));
    }
    values[i] = value;
  }
  local_absmax = BlockReduceMax_ROW<float, BLOCK>(local_absmax, max_shared);
  if (tid == 0) {
    token_absmax = fmaxf(local_absmax, 1.0e-10f);
    scales[row] = token_absmax / 127.0f;
  }
  __syncthreads();
  const float inv_scale = 127.0f / token_absmax;
#pragma unroll
  for (int i = 0; i < ITEMS; ++i) {
    const int64_t offset = tid + i * BLOCK;
    if (offset < hidden_dim) {
      output[row * hidden_dim + offset] =
          static_cast<int8_t>(nearbyintf(values[i] * inv_scale));
    }
  }
}

// The output projection uses 4096 values per token. Keep this common case
// static so address calculations and tail predicates do not dominate gating.
template <typename scalar_t, int BLOCK>
__global__ __launch_bounds__(BLOCK) void fused_sigmoid_mul_quant_4k_vec8_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ gate,
    int8_t* __restrict__ output,
    float* __restrict__ scales,
    int64_t gate_stride_row,
    int64_t gate_stride_head) {
  constexpr int VEC = 8;
  constexpr int CHUNKS = 4096 / (BLOCK * VEC);
  constexpr int WARPS = BLOCK / C10_WARP_SIZE;
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid % C10_WARP_SIZE;
  const int warp = tid / C10_WARP_SIZE;
  __shared__ float warp_maxima[WARPS];
  float values[CHUNKS][VEC];
  float local_max = 0.0f;
  using Pack = at::native::memory::aligned_vector<scalar_t, VEC>;
#pragma unroll
  for (int k = 0; k < CHUNKS; ++k) {
    const int col = (tid + k * BLOCK) * VEC;
    const Pack xx = *reinterpret_cast<const Pack*>(x + row * 4096 + col);
    const Pack gg = *reinterpret_cast<const Pack*>(
        gate + row * gate_stride_row + (col / 256) * gate_stride_head + col % 256);
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
      const float y = static_cast<float>(xx.val[i]) *
          sigmoid_fp32<scalar_t>(static_cast<float>(gg.val[i]));
      const float rounded = static_cast<float>(static_cast<scalar_t>(y));
      values[k][i] = rounded;
      local_max = fmaxf(local_max, fabsf(rounded));
    }
  }
  const float warp_max = WarpReduceMax_ROW<float>(local_max);
  if (lane == 0) warp_maxima[warp] = warp_max;
  __syncthreads();
  float token_max = lane < WARPS ? warp_maxima[lane] : 0.0f;
  token_max = WarpReduceMax_ROW<float, WARPS>(token_max);
#ifdef USE_ROCM
  token_max = __shfl(token_max, 0, C10_WARP_SIZE);
#else
  token_max = __shfl_sync(0xffffffffu, token_max, 0, C10_WARP_SIZE);
#endif
  token_max = fmaxf(token_max, 1e-10f);
  if (tid == 0) scales[row] = token_max / 127.0f;
  const float inv_scale = 127.0f / token_max;
  using OutPack = at::native::memory::aligned_vector<int8_t, VEC>;
#pragma unroll
  for (int k = 0; k < CHUNKS; ++k) {
    OutPack packed;
#pragma unroll
    for (int i = 0; i < VEC; ++i)
      packed.val[i] = static_cast<int8_t>(nearbyintf(values[k][i] * inv_scale));
    const int col = (tid + k * BLOCK) * VEC;
    *reinterpret_cast<OutPack*>(output + row * 4096 + col) = packed;
  }
}

template <typename scalar_t, int HEAD_DIM, int BLOCK>
void launch_cached_sigmoid_quant(
    torch::Tensor& output, const torch::Tensor& attn_output,
    const torch::Tensor& gate, torch::Tensor& scales,
    int64_t hidden_dim, int64_t gate_stride_row,
    int64_t gate_stride_head, int64_t head_dim,
    cudaStream_t stream) {
#define SIGMOID_CACHE_LAUNCH(ITEMS) \
  fused_sigmoid_mul_per_token_quant_cached_kernel<scalar_t, ITEMS, HEAD_DIM, BLOCK> \
      <<<attn_output.size(0), BLOCK, 0, stream>>>( \
          attn_output.data_ptr<scalar_t>(), gate.data_ptr<scalar_t>(), \
          output.data_ptr<int8_t>(), scales.data_ptr<float>(), hidden_dim, \
          gate_stride_row, gate_stride_head, head_dim)
  if (hidden_dim <= BLOCK) { SIGMOID_CACHE_LAUNCH(1); }
  else if (hidden_dim <= BLOCK * 2) { SIGMOID_CACHE_LAUNCH(2); }
  else if (hidden_dim <= BLOCK * 4) { SIGMOID_CACHE_LAUNCH(4); }
  else if constexpr (BLOCK == 256) {
    if (hidden_dim <= 2048) { SIGMOID_CACHE_LAUNCH(8); }
    else if (hidden_dim <= 4096) { SIGMOID_CACHE_LAUNCH(16); }
    else { SIGMOID_CACHE_LAUNCH(32); }
  } else { SIGMOID_CACHE_LAUNCH(8); }
#undef SIGMOID_CACHE_LAUNCH
}

template <typename scalar_t, int HEAD_DIM>
void dispatch_cached_sigmoid_quant(
    torch::Tensor& output, const torch::Tensor& attn_output,
    const torch::Tensor& gate, torch::Tensor& scales,
    int64_t hidden_dim, int64_t gate_stride_row,
    int64_t gate_stride_head, int64_t head_dim,
    cudaStream_t stream) {
  if (attn_output.size(0) <= 512 && hidden_dim >= 1024) {
    launch_cached_sigmoid_quant<scalar_t, HEAD_DIM, 1024>(
        output, attn_output, gate, scales, hidden_dim, gate_stride_row,
        gate_stride_head, head_dim, stream);
  } else {
    launch_cached_sigmoid_quant<scalar_t, HEAD_DIM, 256>(
        output, attn_output, gate, scales, hidden_dim, gate_stride_row,
        gate_stride_head, head_dim, stream);
  }
}

template <typename scalar_t, int block_size>
__global__ __launch_bounds__(block_size) void fused_sigmoid_mul_per_token_quant_kernel(
    const scalar_t* __restrict__ attn_output,
    const scalar_t* __restrict__ gate,
    int8_t* __restrict__ output,
    float* __restrict__ scales,
    int64_t hidden_dim,
    int64_t gate_stride_row,
    int64_t gate_stride_head,
    int64_t head_dim) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int64_t tid = static_cast<int64_t>(threadIdx.x);

  __shared__ float max_shared[block_size / C10_WARP_SIZE];
  __shared__ float token_absmax;

  float local_absmax = 0.0f;
  for (int64_t offset = tid; offset < hidden_dim; offset += block_size) {
    const int64_t head = offset / head_dim;
    const int64_t d = offset - head * head_dim;
    const float x = static_cast<float>(attn_output[row * hidden_dim + offset]);
    const float g = static_cast<float>(
        gate[row * gate_stride_row + head * gate_stride_head + d]);
    // The original SGLang kernel stores the FP32 product into a BF16/FP16
    // tensor before the separate per-token quantization launch. Preserve that
    // intermediate dtype rounding here before computing absmax and INT8.
    const scalar_t rounded = static_cast<scalar_t>(x * sigmoid_fp32<scalar_t>(g));
    const float value = static_cast<float>(rounded);
    local_absmax = fmaxf(local_absmax, fabsf(value));
  }

  local_absmax = BlockReduceMax_ROW<float, block_size>(local_absmax, max_shared);
  if (tid == 0) {
    token_absmax = fmaxf(local_absmax, 1.0e-10f);
    scales[row] = token_absmax / 127.0f;
  }
  __syncthreads();

  const float inv_scale = 127.0f / token_absmax;
  for (int64_t offset = tid; offset < hidden_dim; offset += block_size) {
    const int64_t head = offset / head_dim;
    const int64_t d = offset - head * head_dim;
    const float x = static_cast<float>(attn_output[row * hidden_dim + offset]);
    const float g = static_cast<float>(
        gate[row * gate_stride_row + head * gate_stride_head + d]);
    const scalar_t rounded = static_cast<scalar_t>(x * sigmoid_fp32<scalar_t>(g));
    const float value = static_cast<float>(rounded);
    output[row * hidden_dim + offset] = static_cast<int8_t>(nearbyintf(value * inv_scale));
  }
}

template <typename scalar_t>
void launch_fused_sigmoid_mul_per_token_quant(
    torch::Tensor& output,
    const torch::Tensor& attn_output,
    const torch::Tensor& gate,
    torch::Tensor& scales,
    int64_t hidden_dim,
    int64_t gate_stride_row,
    int64_t gate_stride_head,
    int64_t head_dim) {
  const auto device_guard = at::cuda::OptionalCUDAGuard(device_of(attn_output));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int64_t rows = attn_output.size(0);

  const bool flat_gate = gate.dim() == 2 || gate.size(1) == 1;
  const int64_t packed_head_stride = flat_gate ? 256 : gate_stride_head;
  const bool aligned_4k = hidden_dim == 4096 && (flat_gate || head_dim == 256) &&
      reinterpret_cast<uintptr_t>(attn_output.data_ptr()) % 16 == 0 &&
      reinterpret_cast<uintptr_t>(gate.data_ptr()) % 16 == 0 &&
      reinterpret_cast<uintptr_t>(output.data_ptr()) % 8 == 0 &&
      gate_stride_row % 8 == 0 && packed_head_stride % 8 == 0;
  if (aligned_4k) {
#define SIGMOID_4K_LAUNCH(BLOCK) \
    fused_sigmoid_mul_quant_4k_vec8_kernel<scalar_t, BLOCK> \
        <<<rows, BLOCK, 0, stream>>>( \
            attn_output.data_ptr<scalar_t>(), gate.data_ptr<scalar_t>(), \
            output.data_ptr<int8_t>(), scales.data_ptr<float>(), \
            gate_stride_row, packed_head_stride)
    if (rows <= 32 || rows >= 512) { SIGMOID_4K_LAUNCH(512); }
    else { SIGMOID_4K_LAUNCH(256); }
#undef SIGMOID_4K_LAUNCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }

  // Keep each rounded gate product in registers for the token reduction.
  // Common head dimensions also avoid a runtime integer division per value.
  // Wider rows retain the bounded-register, two-pass implementation below.
  if (hidden_dim <= 8192) {
    if (gate.dim() == 2 || gate.size(1) == 1) {
      dispatch_cached_sigmoid_quant<scalar_t, 0>(
          output, attn_output, gate, scales, hidden_dim, gate_stride_row,
          gate_stride_head, head_dim, stream);
    } else if (head_dim == 128) {
      dispatch_cached_sigmoid_quant<scalar_t, 128>(
          output, attn_output, gate, scales, hidden_dim, gate_stride_row,
          gate_stride_head, head_dim, stream);
    } else if (head_dim == 256) {
      dispatch_cached_sigmoid_quant<scalar_t, 256>(
          output, attn_output, gate, scales, hidden_dim, gate_stride_row,
          gate_stride_head, head_dim, stream);
    } else {
      dispatch_cached_sigmoid_quant<scalar_t, -1>(
          output, attn_output, gate, scales, hidden_dim, gate_stride_row,
          gate_stride_head, head_dim, stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }

  if (hidden_dim <= 256) {
    fused_sigmoid_mul_per_token_quant_kernel<scalar_t, 256>
        <<<rows, 256, 0, stream>>>(
            attn_output.data_ptr<scalar_t>(), gate.data_ptr<scalar_t>(),
            output.data_ptr<int8_t>(), scales.data_ptr<float>(), hidden_dim,
            gate_stride_row, gate_stride_head, head_dim);
  } else if (hidden_dim <= 512) {
    fused_sigmoid_mul_per_token_quant_kernel<scalar_t, 512>
        <<<rows, 512, 0, stream>>>(
            attn_output.data_ptr<scalar_t>(), gate.data_ptr<scalar_t>(),
            output.data_ptr<int8_t>(), scales.data_ptr<float>(), hidden_dim,
            gate_stride_row, gate_stride_head, head_dim);
  } else {
    fused_sigmoid_mul_per_token_quant_kernel<scalar_t, 1024>
        <<<rows, 1024, 0, stream>>>(
            attn_output.data_ptr<scalar_t>(), gate.data_ptr<scalar_t>(),
            output.data_ptr<int8_t>(), scales.data_ptr<float>(), hidden_dim,
            gate_stride_row, gate_stride_head, head_dim);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fused_sigmoid_mul_per_token_quant(
    torch::Tensor& output,
    const torch::Tensor& attn_output,
    const torch::Tensor& gate,
    torch::Tensor& scales) {
  TORCH_CHECK(attn_output.is_cuda(), "attn_output must be a CUDA/HIP tensor");
  TORCH_CHECK(gate.is_cuda() && gate.device() == attn_output.device(),
              "gate must be on the same CUDA/HIP device");
  TORCH_CHECK(attn_output.dim() == 2, "attn_output must be [tokens, hidden]");
  TORCH_CHECK(attn_output.is_contiguous(), "attn_output must be contiguous");
  TORCH_CHECK(attn_output.scalar_type() == at::kBFloat16 ||
                  attn_output.scalar_type() == at::kHalf,
              "attn_output must be BF16 or FP16");
  TORCH_CHECK(gate.scalar_type() == attn_output.scalar_type(),
              "gate and attn_output must have the same dtype");
  TORCH_CHECK(gate.dim() == 2 || gate.dim() == 3,
              "gate must be [tokens, hidden] or [tokens, heads, head_dim]");
  TORCH_CHECK(gate.stride(-1) == 1, "gate's last dimension must be contiguous");

  const int64_t rows = attn_output.size(0);
  const int64_t hidden_dim = attn_output.size(1);
  int64_t head_dim = hidden_dim;
  int64_t gate_stride_row = hidden_dim;
  int64_t gate_stride_head = hidden_dim;
  if (gate.dim() == 3) {
    TORCH_CHECK(gate.size(0) == rows, "gate token dimension mismatch");
    head_dim = gate.size(2);
    TORCH_CHECK(gate.size(1) * head_dim == hidden_dim,
                "gate hidden dimension mismatch");
    gate_stride_row = gate.stride(0);
    gate_stride_head = gate.stride(1);
  } else {
    TORCH_CHECK(gate.sizes() == attn_output.sizes(),
                "flat gate shape must match attn_output");
    gate_stride_row = gate.stride(0);
  }
  TORCH_CHECK(output.is_cuda() && output.device() == attn_output.device() &&
                  output.scalar_type() == at::kChar &&
                  output.sizes() == attn_output.sizes() && output.is_contiguous(),
              "invalid INT8 output tensor");
  TORCH_CHECK(scales.is_cuda() && scales.device() == attn_output.device() &&
                  scales.scalar_type() == at::kFloat && scales.dim() == 2 &&
                  scales.size(0) == rows && scales.size(1) == 1 &&
                  scales.is_contiguous(),
              "invalid per-token scale tensor");

  TORCH_CHECK(hidden_dim > 0, "hidden dimension must be positive");
  if (rows == 0) return;
  if (attn_output.scalar_type() == at::kBFloat16) {
    launch_fused_sigmoid_mul_per_token_quant<at::BFloat16>(
        output, attn_output, gate, scales, hidden_dim, gate_stride_row,
        gate_stride_head, head_dim);
  } else {
    launch_fused_sigmoid_mul_per_token_quant<at::Half>(
        output, attn_output, gate, scales, hidden_dim, gate_stride_row,
        gate_stride_head, head_dim);
  }
}

}  // namespace native
}  // namespace at
