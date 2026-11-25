#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>

namespace {

using Vec16 = uint32_t __attribute__((ext_vector_type(4)));
constexpr int WaveSize = 64;
constexpr int WavesPerBlock = 4;

template<bool ToNK>
__global__ __launch_bounds__(WaveSize * WavesPerBlock) void pi32_relayout(
    const Vec16* __restrict__ src,
    Vec16* __restrict__ dst,
    int n,
    int k)
{
    const unsigned lane = threadIdx.x % WaveSize;
    const unsigned n16 = blockIdx.y * WavesPerBlock + threadIdx.x / WaveSize;
    if (n16 >= static_cast<unsigned>(n / 16)) {
        return;
    }

    const unsigned permuted_row = n16 * 16 + lane % 16;
    const unsigned i = permuted_row % 32;
    const unsigned row = (permuted_row / 32) * 32 + 8 * (i % 4) + 4 * (i / 16) + (i % 16) / 4;
    const unsigned k64 = blockIdx.x;
    const unsigned k16 = lane / 16;
    const size_t nk_index = static_cast<size_t>(row) * (k / 16) + k64 * 4 + k16;
    const size_t packed_index = (static_cast<size_t>(k64) * (n / 16) + n16) * WaveSize + lane;

    if constexpr (ToNK) {
        dst[nk_index] = src[packed_index];
    } else {
        dst[packed_index] = src[nk_index];
    }
}

}  // namespace

extern "C" int pi32_relayout_launch(
    const void* source,
    void* destination,
    int n,
    int k,
    int to_nk,
    hipStream_t stream)
{
    if (n <= 0 || n % 32 || k <= 0 || k % 64 || !source || !destination ||
        source == destination || (reinterpret_cast<uintptr_t>(source) % 16) ||
        (reinterpret_cast<uintptr_t>(destination) % 16)) {
        return hipErrorInvalidValue;
    }
    const auto* src = static_cast<const Vec16*>(source);
    auto* dst = static_cast<Vec16*>(destination);
    const dim3 grid(k / 64, (n / 16 + WavesPerBlock - 1) / WavesPerBlock);
    const dim3 block(WaveSize * WavesPerBlock);
    if (to_nk) {
        hipLaunchKernelGGL((pi32_relayout<true>), grid, block, 0, stream, src, dst, n, k);
    } else {
        hipLaunchKernelGGL((pi32_relayout<false>), grid, block, 0, stream, src, dst, n, k);
    }
    return hipGetLastError();
}
