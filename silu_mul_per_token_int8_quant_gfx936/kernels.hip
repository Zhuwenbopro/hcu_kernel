#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <cstdint>

// Build each measured mapping in a separate device code object to preserve
// the input-load schedule, kernel placement and position of its 128 KiB LUT.
namespace {
constexpr int Wave = 64;
#if defined(SILU_PAIR4)
constexpr int Vec = 4;
#else
constexpr int Vec = 8;
#endif
constexpr float Log2E = 1.44269504088896340736f;
using Load = uint32_t __attribute__((ext_vector_type(4), may_alias));

#if defined(SILU_PAIR4)
struct alignas(4) Int8Pack { int8_t values[Vec]; };
struct Tiles { Load pairs[2]; };

#else
struct Bf16Pack { uint16_t bits[Vec]; };
struct alignas(8) Int8Pack { int8_t values[Vec]; };
struct Tiles { Load gate[2]; Load up[2]; };

#endif
__device__ __forceinline__ float from_bf16(uint16_t bits) {
    return __uint_as_float(uint32_t(bits) << 16);
}

__device__ __forceinline__ float to_bf16_and_back(float value) {
    return static_cast<float>(hip_bfloat16(value));
}

__device__ uint16_t gate_lut[1 << 16];

__global__ void initialize_gate_lut() {
    const unsigned bits = blockIdx.x * blockDim.x + threadIdx.x;
    const float x = from_bf16(bits);
    const float denominator = 1.0f + __builtin_amdgcn_exp2f(-x * Log2E);
    const float silu = to_bf16_and_back(x / denominator);
    gate_lut[bits] = __float_as_uint(silu) >> 16;
    return;
}

#if defined(SILU_PAIR4)
__device__ __forceinline__ float activate(Load pairs, float (&values)[Vec]) {
#else
__device__ __forceinline__ float activate(Load gate, Load up, float (&values)[Vec]) {
#if defined(SILU_PAIR8)
    // Each dword contains one pair, with gate in bits 0..15 and up in 16..31.
    const Load interleaved[2] = {gate, up};
    Bf16Pack g, u;
    #pragma unroll
    for (int i = 0; i < Vec; ++i) {
        const uint32_t pair = interleaved[i / 4][i % 4];
        g.bits[i] = static_cast<uint16_t>(pair);
        u.bits[i] = static_cast<uint16_t>(pair >> 16);
    }
#else
    const auto g = __builtin_bit_cast(Bf16Pack, gate);
    const auto u = __builtin_bit_cast(Bf16Pack, up);
#endif
#endif
    uint16_t silu[Vec];
    #pragma unroll
    for (int i = 0; i < Vec; ++i) {
#if defined(SILU_PAIR4)
        silu[i] = gate_lut[static_cast<uint16_t>(pairs[i])];
#else
        silu[i] = gate_lut[g.bits[i]];
#endif
    }
    float row_max = 0.0f;
    #pragma unroll
    for (int i = 0; i < Vec; ++i) {
#if defined(SILU_PAIR4)
        const float result = to_bf16_and_back(from_bf16(silu[i]) * from_bf16(static_cast<uint16_t>(pairs[i] >> 16)));
#else
        const float result = to_bf16_and_back(from_bf16(silu[i]) * from_bf16(u.bits[i]));
#endif
        values[i] = result;
        row_max = fmaxf(row_max, fabsf(result));
    }
    return row_max;
}

#if defined(SILU_PAIR4)
template <bool Coherent>
__device__ __forceinline__ Tiles issue(const Load* input, unsigned offset, unsigned next) {
    Tiles data;
#define LOAD_PAIR(FLAGS) \
    asm volatile( \
        "global_load_dwordx4 %0, %2, %4" FLAGS "\n\t" \
        "global_load_dwordx4 %1, %3, %4" FLAGS "\n\t" \
        "s_waitcnt vmcnt(1)" \
        : "=&v"(data.pairs[0]), "=&v"(data.pairs[1]) \
        : "v"(offset), "v"(next), "s"(input) : "memory")
    if constexpr (Coherent) {
        LOAD_PAIR(" glc slc");
    } else {
        LOAD_PAIR("");
    }
#undef LOAD_PAIR
    return data;
}

#else
template <bool Coherent>
__device__ __forceinline__ Tiles issue(const Load* gate, const Load* up,
                                      unsigned offset, unsigned next) {
    Tiles data;
#define LOAD_PAIR(FLAGS) \
    asm volatile( \
        "global_load_dwordx4 %0, %4, %6" FLAGS "\n\t" \
        "global_load_dwordx4 %1, %4, %7" FLAGS "\n\t" \
        "global_load_dwordx4 %2, %5, %6" FLAGS "\n\t" \
        "global_load_dwordx4 %3, %5, %7" FLAGS "\n\t" \
        "s_waitcnt vmcnt(2)" \
        : "=&v"(data.gate[0]), "=&v"(data.up[0]), "=&v"(data.gate[1]), "=&v"(data.up[1]) \
        : "v"(offset), "v"(next), "s"(gate), "s"(up) : "memory")
    if constexpr (Coherent) {
        LOAD_PAIR(" glc slc");
    } else {
        LOAD_PAIR("");
    }
#undef LOAD_PAIR
    return data;
}

#endif
template <int Offset>
__device__ __forceinline__ float reduce_max(float value) {
    if constexpr (Offset > 0) {
        value = fmaxf(value, __shfl_down(value, Offset, Wave));
        return reduce_max<Offset / 2>(value);
    }
    return value;
}

template <int D, int Threads, int Mode>
__global__ __launch_bounds__(Threads) void silu_quant(
    const Load* __restrict__ input, Int8Pack* __restrict__ output, float* __restrict__ scales) {
    constexpr int Packs = D / Vec;
    constexpr int FullTiles = Packs / Threads;
    constexpr int TilesPerThread = (Packs + Threads - 1) / Threads;
    constexpr int Waves = Threads / Wave;
    constexpr int ReduceSlots = Waves <= 2 ? 2 : Waves <= 4 ? 4 : Waves <= 8 ? 8 : 16;
    constexpr int Schedule = Mode % 4;
    constexpr bool Staged = Schedule == 1 || Schedule == 2;
    __shared__ float partials[Waves];
    __shared__ float broadcast;

    const int tid = threadIdx.x;
    const int lane = tid % Wave;
    const int wave = tid / Wave;
#if defined(SILU_PAIR4)
    const auto* pairs = input + int64_t(blockIdx.x) * Packs;
#else
    const auto* gate = input + int64_t(blockIdx.x) * 2 * Packs;
#if defined(SILU_PAIR8)
    const auto* up = gate + 1;
#else
    const auto* up = gate + Packs;
#endif
#endif
    auto* out = output + int64_t(blockIdx.x) * Packs;
    float values[TilesPerThread][Vec];
    float row_max = 0.0f;
    constexpr int PairedTiles = Staged ? (FullTiles / 2) * 2 : 0;

    #pragma unroll
    for (int tile = 0; tile < PairedTiles; tile += 2) {
#if defined(SILU_PAIR4)
        const unsigned offset = (tid + tile * Threads) * sizeof(Load);
        auto data = issue<Schedule == 2>(pairs, offset, offset + Threads * sizeof(Load));
        const float first_max = activate(data.pairs[0], values[tile]);
        asm volatile("s_waitcnt vmcnt(0)"
                     : "+v"(data.pairs[1]) : "v"(first_max) : "memory");
        const float second_max = activate(data.pairs[1], values[tile + 1]);
#else
#if defined(SILU_PAIR8)
        const unsigned offset = (tid + tile * Threads) * (2 * sizeof(Load));
        auto data = issue<Schedule == 2>(gate, up, offset, offset + Threads * (2 * sizeof(Load)));
#else
        const unsigned offset = (tid + tile * Threads) * sizeof(Load);
        auto data = issue<Schedule == 2>(gate, up, offset, offset + Threads * sizeof(Load));
#endif
        const float first_max = activate(data.gate[0], data.up[0], values[tile]);
        asm volatile("s_waitcnt vmcnt(0)"
                     : "+v"(data.gate[1]), "+v"(data.up[1]) : "v"(first_max) : "memory");
        const float second_max = activate(data.gate[1], data.up[1], values[tile + 1]);
#endif
        row_max = fmaxf(row_max, fmaxf(first_max, second_max));
    }

    #pragma unroll
    for (int tile = PairedTiles; tile < TilesPerThread; ++tile) {
        const int column = tid + tile * Threads;
        if (column < Packs) {
#if defined(SILU_PAIR4)
            row_max = fmaxf(row_max, activate(pairs[column], values[tile]));
#else
#if defined(SILU_PAIR8)
            row_max = fmaxf(row_max, activate(gate[2 * column], up[2 * column], values[tile]));
#else
            row_max = fmaxf(row_max, activate(gate[column], up[column], values[tile]));
#endif
#endif
        }
    }

    row_max = reduce_max<Wave / 2>(row_max);
    if (lane == 0) {
        partials[wave] = row_max;
    }
    __syncthreads();

    if constexpr (Schedule == 3) {
        if (wave == 0) {
            const float partial = lane < Waves ? partials[lane] : 0.0f;
            const float combined = reduce_max<ReduceSlots / 2>(partial);
            if (lane == 0) {
                broadcast = combined;
            }
        }
        __syncthreads();
        row_max = broadcast;
    } else {
        const float partial = lane < Waves ? partials[lane] : 0.0f;
        row_max = __shfl(reduce_max<ReduceSlots / 2>(partial), 0, Wave);
    }
    if (tid == 0) {
        scales[blockIdx.x] = row_max / 127.0f;
    }
    const float inverse = row_max == 0.0f ? 0.0f : 127.0f / row_max;

    #pragma unroll
    for (int tile = 0; tile < TilesPerThread; ++tile) {
        const int column = tid + tile * Threads;
        if (column < Packs) {
            Int8Pack quantized;
            #pragma unroll
            for (int i = 0; i < Vec; ++i) {
                float q = nearbyintf(values[tile][i] * inverse);
                q = q < -128.0f ? -128.0f : q > 127.0f ? 127.0f : q;
                quantized.values[i] = static_cast<int8_t>(q);
            }
            out[column] = quantized;
        }
    }
    return;
}

template <int D, int Threads>
hipError_t dispatch_mode(int mode, int rows, const void* input, void* output,
                         float* scale, hipStream_t stream) {
#define LAUNCH(MODE) \
    case MODE: silu_quant<D, Threads, MODE><<<rows, Threads, 0, stream>>>( \
        static_cast<const Load*>(input), static_cast<Int8Pack*>(output), scale); break
    switch (mode) {
        LAUNCH(0);
        LAUNCH(1);
        LAUNCH(2);
        LAUNCH(3);
        default: return hipErrorInvalidValue;
    }
#undef LAUNCH
    return hipGetLastError();
}

template <int D>
hipError_t dispatch_threads(int threads, int mode, int rows, const void* input,
                            void* output, float* scale, hipStream_t stream) {
#define DISPATCH(THREADS) \
    case THREADS: return dispatch_mode<D, THREADS>(mode, rows, input, output, scale, stream)
    switch (threads) {
        DISPATCH(256);
        DISPATCH(320);
        DISPATCH(512);
        DISPATCH(768);
    }
#undef DISPATCH
    return hipErrorInvalidValue;
}
} // namespace

extern "C" hipError_t launch_silu_quant(int d, int threads, int mode, int rows,
                                        const void* input, void* output, float* scale,
                                        hipStream_t stream) {
    if (rows == 0) return hipSuccess;
#define DISPATCH(D) case D: return dispatch_threads<D>(threads, mode, rows, input, output, scale, stream)
    switch (d) {
#if defined(SILU_TP2)
        DISPATCH(8704);
#else
        DISPATCH(5120);
        DISPATCH(17408);
#endif
    }
#undef DISPATCH
    return hipErrorInvalidValue;
}

extern "C" hipError_t init_silu_gate_lut(hipStream_t stream) {
    initialize_gate_lut<<<256, 256, 0, stream>>>();
    return hipGetLastError();
}
