#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace {

constexpr int Width = 5120;
constexpr int Vec = 8;
constexpr int WaveSize = 64;
constexpr int PacksPerRow = Width / Vec;
constexpr int WavesPerRow = PacksPerRow / WaveSize;
constexpr int StagedThreads = PacksPerRow / 2;
constexpr int ReduceSlots = 16;
constexpr float MinAbsmax = 1.0e-10f;
constexpr float Qmax = 127.0f;

struct alignas(16) Bf16Pack {
    uint16_t bits[Vec];
};

struct alignas(8) Int8Pack {
    int8_t values[Vec];
};

using Dwordx4 = uint32_t __attribute__((ext_vector_type(4), may_alias));

enum class Loads { Compiler, Staged, StagedGlcSlc };

struct TwoTiles {
    Dwordx4 x[2];
    Dwordx4 r[2];
    Dwordx4 w[2];
};

template <Loads Mode>
__device__ __forceinline__ TwoTiles load_tiles(
    const Bf16Pack* x, const Bf16Pack* residual, const Bf16Pack* weight, unsigned offset) {
    TwoTiles data;
    const unsigned next = offset + StagedThreads * sizeof(Bf16Pack);
    // Issue three vector loads per tile, then wait for the first tile.
#define LOAD_TILES(FLAGS)                                                            \
    asm volatile(                                                                    \
        "global_load_dwordx4 %0, %6, %8" FLAGS "\n\t"                                  \
        "global_load_dwordx4 %1, %6, %9" FLAGS "\n\t"                                  \
        "global_load_dwordx4 %2, %6, %10" FLAGS "\n\t"                                 \
        "global_load_dwordx4 %3, %7, %8" FLAGS "\n\t"                                  \
        "global_load_dwordx4 %4, %7, %9" FLAGS "\n\t"                                  \
        "global_load_dwordx4 %5, %7, %10" FLAGS "\n\t"                                 \
        "s_waitcnt vmcnt(3)"                                                          \
        : "=&v"(data.x[0]), "=&v"(data.r[0]), "=&v"(data.w[0]),                        \
          "=&v"(data.x[1]), "=&v"(data.r[1]), "=&v"(data.w[1])                         \
        : "v"(offset), "v"(next), "s"(x), "s"(residual), "s"(weight) : "memory")
    if constexpr (Mode == Loads::StagedGlcSlc) {
        LOAD_TILES(" glc slc");
    } else {
        LOAD_TILES("");
    }
#undef LOAD_TILES
    return data;
}

__device__ __forceinline__ void wait_for_next_tile(TwoTiles& data, float2 previous) {
    // Depend on the first tile's statistics so this wait follows its computation.
    asm volatile("s_waitcnt vmcnt(0)" : "+v"(data.x[1]), "+v"(data.r[1]), "+v"(data.w[1])
                 : "v"(previous.x), "v"(previous.y) : "memory");
    return;
}

template <bool ForceVector>
__device__ __forceinline__ Bf16Pack load_pack(const Bf16Pack* ptr) {
    if constexpr (ForceVector) {
        return __builtin_bit_cast(Bf16Pack, *reinterpret_cast<const Dwordx4*>(ptr));
    }
    return *ptr;
}

__device__ __forceinline__ float as_float(uint16_t bits) {
    return __uint_as_float(static_cast<uint32_t>(bits) << 16);
}

__device__ __forceinline__ float2 local_stats(
    const Bf16Pack& x, const Bf16Pack& residual, const Bf16Pack& weight,
    float (&weighted)[Vec], Bf16Pack& updated) {
    float2 stats = make_float2(0.0f, 0.0f);
#pragma unroll
    for (int i = 0; i < Vec; ++i) {
        const float sum = as_float(x.bits[i]) + as_float(residual.bits[i]);
        stats.x += sum * sum;
        const float value = sum * (1.0f + as_float(weight.bits[i]));
        weighted[i] = value;
        stats.y = fmaxf(stats.y, fabsf(value));
        updated.bits[i] = hip_bfloat16(sum).data;
    }
    return stats;
}

template <int Offset>
__device__ __forceinline__ float shuffle_down(float value) {
    if constexpr (Offset >= 16) {
        return __shfl_down(value, Offset, WaveSize);
    } else {
        // Only the lower lane of each DPP row keeps the reduction result.
        constexpr int RowShiftLeft = 0x100;
        return __uint_as_float(__builtin_amdgcn_mov_dpp(
            __float_as_uint(value), RowShiftLeft + Offset, 0xf, 0xf, true));
    }
}

template <int Offset>
__device__ __forceinline__ float2 reduce(float2 stats) {
    if constexpr (Offset > 0) {
        stats.x += shuffle_down<Offset>(stats.x);
        stats.y = fmaxf(stats.y, shuffle_down<Offset>(stats.y));
        return reduce<Offset / 2>(stats);
    }
    return stats;
}

__device__ __forceinline__ void publish_wave(float2* partials, int slot, int lane, float2 local) {
    const float2 stats = reduce<WaveSize / 2>(local);
    if (lane == 0) {
        partials[slot] = stats;
    }
    return;
}

template <int Threads, Loads Mode>
__global__ __launch_bounds__(Threads) void gemma_int8_5120(
    const Bf16Pack* __restrict__ x, Bf16Pack* __restrict__ residual,
    const Bf16Pack* __restrict__ weight, Int8Pack* __restrict__ out,
    float* __restrict__ scales, float eps) {
    constexpr int Groups = (PacksPerRow + Threads - 1) / Threads;
    constexpr int Waves = Threads / WaveSize;
    static_assert(Threads % WaveSize == 0 && PacksPerRow % WaveSize == 0);
    static_assert(WavesPerRow <= ReduceSlots);
    static_assert(Mode == Loads::Compiler || Threads == StagedThreads);

    __shared__ float2 partials[ReduceSlots];
    const int tid = threadIdx.x;
    const int lane = tid % WaveSize;
    const int wave = tid / WaveSize;
    const unsigned row = blockIdx.x;
    const auto base = static_cast<uint64_t>(row) * PacksPerRow;
    float weighted[Groups][Vec];
    if (tid < ReduceSlots - WavesPerRow) {
        partials[WavesPerRow + tid] = make_float2(0.0f, 0.0f);
    }

    if constexpr (Mode == Loads::Compiler) {
        constexpr bool ForceVector = Threads == StagedThreads;
#pragma unroll
        for (int group = 0; group < Groups; ++group) {
            const int col = tid + group * Threads;
            if (col >= PacksPerRow) {
                continue;
            }
            Bf16Pack updated;
            const float2 stats = local_stats(
                load_pack<ForceVector>(x + base + col), load_pack<ForceVector>(residual + base + col),
                load_pack<ForceVector>(weight + col), weighted[group], updated);
            residual[base + col] = updated;
            publish_wave(partials, group * Waves + wave, lane, stats);
        }
    } else {
        TwoTiles data = load_tiles<Mode>(x + base, residual + base, weight, tid * sizeof(Bf16Pack));
        Bf16Pack r0, r1;
        const float2 a = local_stats(__builtin_bit_cast(Bf16Pack, data.x[0]),
                                     __builtin_bit_cast(Bf16Pack, data.r[0]),
                                     __builtin_bit_cast(Bf16Pack, data.w[0]), weighted[0], r0);
        wait_for_next_tile(data, a);
        const float2 b = local_stats(__builtin_bit_cast(Bf16Pack, data.x[1]),
                                     __builtin_bit_cast(Bf16Pack, data.r[1]),
                                     __builtin_bit_cast(Bf16Pack, data.w[1]), weighted[1], r1);
        residual[base + tid] = r0;
        residual[base + tid + Threads] = r1;
        publish_wave(partials, wave, lane, a);
        publish_wave(partials, Waves + wave, lane, b);
    }

    // Reduce both statistics together after one CTA barrier.
    __syncthreads();
    const float2 local = lane < ReduceSlots ? partials[lane] : make_float2(0.0f, 0.0f);
    const float2 reduced = reduce<ReduceSlots / 2>(local);
    const float square_sum = __shfl(reduced.x, 0, WaveSize);
    const float max_weighted = __shfl(reduced.y, 0, WaveSize);
    const float rstd = rsqrtf(square_sum / static_cast<float>(Width) + eps);
    const float absmax = fmaxf(max_weighted * rstd, MinAbsmax);
    if (tid == 0) {
        scales[row] = absmax / Qmax;
    }
    const float inverse = Qmax / absmax;

#pragma unroll
    for (int group = 0; group < Groups; ++group) {
        const int col = tid + group * Threads;
        if (col >= PacksPerRow) {
            continue;
        }
        Int8Pack quantized;
#pragma unroll
        for (int i = 0; i < Vec; ++i) {
            const float normalized = weighted[group][i] * rstd;
            quantized.values[i] = static_cast<int8_t>(nearbyintf(normalized * inverse));
        }
        out[base + col] = quantized;
    }
    return;
}

template <int Threads, Loads Mode = Loads::Compiler>
hipError_t launch(const void* x, void* residual, const void* weight, void* out, float* scales,
                  int tokens, float eps, hipStream_t stream) {
    gemma_int8_5120<Threads, Mode><<<tokens, Threads, 0, stream>>>(
        static_cast<const Bf16Pack*>(x), static_cast<Bf16Pack*>(residual),
        static_cast<const Bf16Pack*>(weight), static_cast<Int8Pack*>(out), scales, eps);
    return hipGetLastError();
}

}  // namespace

extern "C" hipError_t launch_gemma_int8_5120(
    const void* x, void* residual, const void* weight, void* out,
    float* scales, int tokens, float eps, hipStream_t stream) {
    if (tokens <= 0) {
        return tokens == 0 ? hipSuccess : hipErrorInvalidValue;
    }
    if (tokens <= 63) {
        return launch<640>(x, residual, weight, out, scales, tokens, eps, stream);
    }
    if (tokens <= 159 || (tokens >= 704 && tokens <= 1407)) {
        return launch<512>(x, residual, weight, out, scales, tokens, eps, stream);
    }
    if (tokens <= 351) {
        return launch<StagedThreads>(x, residual, weight, out, scales, tokens, eps, stream);
    }
    if (tokens <= 703) {
        return launch<StagedThreads, Loads::StagedGlcSlc>(x, residual, weight, out, scales, tokens, eps, stream);
    }
    return launch<StagedThreads, Loads::Staged>(x, residual, weight, out, scales, tokens, eps, stream);
}
