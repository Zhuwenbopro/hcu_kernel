#include "silu_mul_int8.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <iterator>

namespace {
constexpr int Wave = 64;
constexpr float Log2E = 1.44269504088896340736f;
using Load = uint32_t __attribute__((ext_vector_type(4), may_alias));

enum class Layout { Split8, Pair4, Pair8 };
template <Layout Kind>
constexpr int Width = Kind == Layout::Pair4 ? 4 : 8;

struct Bf16Pack { uint16_t bits[8]; };
template <int Vec>
struct alignas(Vec) Int8Pack { int8_t values[Vec]; };

struct EightTiles { Load gate[2]; Load up[2]; };
struct FourTiles { Load pairs[2]; };

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

// A split tile holds eight gates in the first vector and eight ups in the second.
// A pair tile holds four [gate, up] BF16 pairs in each 16-byte vector.
template <Layout Kind>
__device__ __forceinline__ float activate_eight(Load first, Load second, float (&values)[8]) {
    constexpr int Vec = 8;
    Bf16Pack g, u;
    if constexpr (Kind == Layout::Split8) {
        g = __builtin_bit_cast(Bf16Pack, first);
        u = __builtin_bit_cast(Bf16Pack, second);
    } else {
        const Load interleaved[2] = {first, second};
        #pragma unroll
        for (int i = 0; i < Vec; ++i) {
            const uint32_t pair = interleaved[i / 4][i % 4];
            g.bits[i] = static_cast<uint16_t>(pair);
            u.bits[i] = static_cast<uint16_t>(pair >> 16);
        }
    }
    uint16_t silu[Vec];
    #pragma unroll
    for (int i = 0; i < Vec; ++i) {
        silu[i] = gate_lut[g.bits[i]];
    }
    float row_max = 0.0f;
    #pragma unroll
    for (int i = 0; i < Vec; ++i) {
        const float result = to_bf16_and_back(from_bf16(silu[i]) * from_bf16(u.bits[i]));
        values[i] = result;
        row_max = fmaxf(row_max, fabsf(result));
    }
    return row_max;
}

__device__ __forceinline__ float activate_four(Load pairs, float (&values)[4]) {
    constexpr int Vec = 4;
    uint16_t silu[Vec];
    #pragma unroll
    for (int i = 0; i < Vec; ++i) {
        silu[i] = gate_lut[static_cast<uint16_t>(pairs[i])];
    }
    float row_max = 0.0f;
    #pragma unroll
    for (int i = 0; i < Vec; ++i) {
        const float result = to_bf16_and_back(
            from_bf16(silu[i]) * from_bf16(static_cast<uint16_t>(pairs[i] >> 16)));
        values[i] = result;
        row_max = fmaxf(row_max, fabsf(result));
    }
    return row_max;
}

// Issue two tiles before activating the first. Keep the second tile in flight.
template <bool Coherent>
__device__ __forceinline__ EightTiles issue_eight(const Load* gate, const Load* up,
                                      unsigned offset, unsigned next) {
    EightTiles data;
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

template <bool Coherent>
__device__ __forceinline__ FourTiles issue_four(const Load* input, unsigned offset, unsigned next) {
    FourTiles data;
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

template <int Offset>
__device__ __forceinline__ float reduce_max(float value) {
    if constexpr (Offset > 0) {
        value = fmaxf(value, __shfl_down(value, Offset, Wave));
        return reduce_max<Offset / 2>(value);
    }
    return value;
}

// One block reduces and quantizes one row. Values stay in registers until
// every wave has contributed its maximum to the row scale.
template <int D, int Threads, int Schedule, Layout Kind>
__global__ __launch_bounds__(Threads) void silu_quant(
    const Load* __restrict__ input, Int8Pack<Width<Kind>>* __restrict__ output,
    float* __restrict__ scales) {
    constexpr int Vec = Width<Kind>;
    constexpr int Packs = D / Vec;
    constexpr int FullTiles = Packs / Threads;
    constexpr int TilesPerThread = (Packs + Threads - 1) / Threads;
    constexpr int Waves = Threads / Wave;
    constexpr int ReduceSlots = Waves <= 2 ? 2 : Waves <= 4 ? 4 : Waves <= 8 ? 8 : 16;
    constexpr bool Staged = Schedule == 1 || Schedule == 2;
    __shared__ float partials[Waves];
    __shared__ float broadcast;

    const int tid = threadIdx.x;
    const int lane = tid % Wave;
    const int wave = tid / Wave;
    constexpr int Stride = Kind == Layout::Pair8 ? 2 : 1;
    const Load* row = input + int64_t(blockIdx.x) * (2 * D / 8);
    const Load* up = row + (Kind == Layout::Split8 ? Packs : 1);
    auto* out = output + int64_t(blockIdx.x) * Packs;
    float values[TilesPerThread][Vec];
    float row_max = 0.0f;
    constexpr int PairedTiles = Staged ? (FullTiles / 2) * 2 : 0;

    #pragma unroll
    for (int tile = 0; tile < PairedTiles; tile += 2) {
        const unsigned offset = (tid + tile * Threads) * Stride * sizeof(Load);
        const unsigned next = offset + Threads * Stride * sizeof(Load);
        float first_max, second_max;
        if constexpr (Kind == Layout::Pair4) {
            auto data = issue_four<Schedule == 2>(row, offset, next);
            first_max = activate_four(data.pairs[0], values[tile]);
            asm volatile("s_waitcnt vmcnt(0)"
                         : "+v"(data.pairs[1]) : "v"(first_max) : "memory");
            second_max = activate_four(data.pairs[1], values[tile + 1]);
        } else {
            auto data = issue_eight<Schedule == 2>(row, up, offset, next);
            first_max = activate_eight<Kind>(data.gate[0], data.up[0], values[tile]);
            asm volatile("s_waitcnt vmcnt(0)"
                         : "+v"(data.gate[1]), "+v"(data.up[1]) : "v"(first_max) : "memory");
            second_max = activate_eight<Kind>(data.gate[1], data.up[1], values[tile + 1]);
        }
        row_max = fmaxf(row_max, fmaxf(first_max, second_max));
    }

    #pragma unroll
    for (int tile = PairedTiles; tile < TilesPerThread; ++tile) {
        const int column = tid + tile * Threads;
        if (column < Packs) {
            if constexpr (Kind == Layout::Pair4) {
                row_max = fmaxf(row_max, activate_four(row[column], values[tile]));
            } else {
                row_max = fmaxf(row_max, activate_eight<Kind>(
                    row[Stride * column], up[Stride * column], values[tile]));
            }
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
            Int8Pack<Vec> quantized;
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

using Launcher = hipError_t (*)(int, const void*, int8_t*, float*, hipStream_t);

template <int D, Layout Kind, int Threads, int Schedule>
hipError_t launch(int rows, const void* input, int8_t* output, float* scales, hipStream_t stream) {
    silu_quant<D, Threads, Schedule, Kind><<<rows, Threads, 0, stream>>>(
        static_cast<const Load*>(input), reinterpret_cast<Int8Pack<Width<Kind>>*>(output), scales);
    return hipGetLastError();
}

struct Route {
    int last_row_count;
    Launcher split;
    Launcher interleaved;
};

#include "routes.inc"

template <size_t N>
const Route& select_route(const Route (&routes)[N], int m) {
    return *std::lower_bound(std::begin(routes), std::end(routes), m,
        [](const Route& route, int rows) { return route.last_row_count < rows; });
}

hipError_t dispatch(bool interleaved, int m, int d, const void* input, int8_t* output,
                    float* scales, hipStream_t stream) {
    if (m < 0) return hipErrorInvalidValue;
    const Route* route;
    switch (d) {
        case 5120:  route = &select_route(routes5120, m); break;
        case 8704:  route = &select_route(routes8704, m); break;
        case 17408: route = &select_route(routes17408, m); break;
        default: return hipErrorInvalidValue;
    }
    if (m == 0) return hipSuccess;
    if (!input || !output || !scales || reinterpret_cast<uintptr_t>(input) % alignof(Load) ||
        reinterpret_cast<uintptr_t>(output) % 8 || reinterpret_cast<uintptr_t>(scales) % alignof(float)) {
        return hipErrorInvalidValue;
    }
    return (interleaved ? route->interleaved : route->split)(m, input, output, scales, stream);
}
} // namespace

extern "C" hipError_t silu_mul_int8_init(hipStream_t stream) {
    initialize_gate_lut<<<256, 256, 0, stream>>>();
    return hipGetLastError();
}

extern "C" hipError_t silu_mul_int8_split(int m, int d, const void* input, int8_t* output,
                                          float* scales, hipStream_t stream) {
    return dispatch(false, m, d, input, output, scales, stream);
}

extern "C" hipError_t silu_mul_int8_interleaved(int m, int d, const void* input, int8_t* output,
                                                float* scales, hipStream_t stream) {
    return dispatch(true, m, d, input, output, scales, stream);
}
