#include <hip/hip_runtime.h>
#include <cstdint>
#include <limits>
#include <type_traits>

using I2 = int32_t __attribute__((ext_vector_type(2)));
using I4 = int32_t __attribute__((ext_vector_type(4)));
using F4 = float __attribute__((ext_vector_type(4)));

#ifdef EXP4_VEC_REDUCE
#define exp4_ring_gemm exp4_ring_vec_gemm
#define exp4_ring_launch exp4_ring_vec_launch
#endif

#ifndef EXP4_NBW
#define EXP4_NBW 4
#endif
static_assert(EXP4_NBW == 2 || EXP4_NBW == 4);
constexpr int TileColumns = 4 * EXP4_NBW * 16;

template<int I, int End, typename Fn>
__device__ __forceinline__ void unroll(Fn&& fn)
{
    if constexpr (I < End) {
        fn(std::integral_constant<int, I>{});
        unroll<I + 1, End>(static_cast<Fn&&>(fn));
    }
}

__device__ __forceinline__ I4 resource(const uint8_t* data)
{
    const uint64_t p = reinterpret_cast<uint64_t>(data);
    return I4{
        __builtin_amdgcn_readfirstlane(static_cast<int>(p)),
        __builtin_amdgcn_readfirstlane(static_cast<int>(p >> 32)),
        -1, 0x00027000
    };
}

template<int Immediate, bool NoL2>
__device__ __forceinline__ void load_16(I4& value, I4 desc, int lane_offset, int block_offset)
{
    const int soffset = __builtin_amdgcn_readfirstlane(block_offset);
    if constexpr (NoL2) {
        asm volatile("buffer_load_dwordx4 %0, %1, %2, %3 offen offset:%4 glc slc"
                     : "+v"(value) : "v"(lane_offset), "s"(desc), "s"(soffset), "n"(Immediate) : "memory");
    } else {
        asm volatile("buffer_load_dwordx4 %0, %1, %2, %3 offen offset:%4"
                     : "+v"(value) : "v"(lane_offset), "s"(desc), "s"(soffset), "n"(Immediate) : "memory");
    }
}

template<int Remain>
__device__ __forceinline__ void wait_vmem()
{
    asm volatile("s_waitcnt vmcnt(%0)" :: "n"(Remain) : "memory");
}

__device__ __forceinline__ void materialize(I4& value)
{
    asm volatile("" : "+v"(value));
}

__device__ __forceinline__ void sync_lds()
{
    __builtin_amdgcn_s_waitcnt(0xc07f);
    asm volatile("s_barrier" ::: "memory");
}

__device__ __forceinline__ F4 mmac(I2 a, I2 b, F4 acc)
{
    return __builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts(a, b, acc, false, false);
}

__device__ __forceinline__ uint32_t bf16_bits(float value)
{
    const uint32_t u = __float_as_uint(value);
    return (u + 0x7fff + ((u >> 16) & 1)) >> 16;
}

__device__ __forceinline__ uint32_t bf16_pair(float left, float right)
{
    return bf16_bits(left) | (bf16_bits(right) << 16);
}

template<int MB, int D, bool Partitioned, bool NoL2>
__global__ __launch_bounds__(256, 1) void exp4_ring_gemm(
    const uint8_t* __restrict__ x, const uint8_t* __restrict__ weight,
    const float* __restrict__ sx, const float* __restrict__ sw,
    uint16_t* __restrict__ y, float* __restrict__ partial,
    int M, int N, int K, int splits)
{
    constexpr int NBW = EXP4_NBW;
    constexpr int BN = TileColumns;
    constexpr int Ahead = 3;
    constexpr int Slots = 4;
    constexpr int XLoads = (MB + 3) / 4;
    constexpr int XSlot = XLoads * 256;
    constexpr int Group = NBW + XLoads;
    static_assert(D == 2 || D == 4);
    __shared__ I4 lds[Slots * XSlot];
    __shared__ float shared_sx[MB * 16];
    __shared__ float shared_sw[BN];

    const int tid = threadIdx.x;
    const int lane = tid & 63;
    const int wave = tid / 64;
    const int col = blockIdx.x * BN;
    const int waves_n = wave * NBW * 16;
    const int partition = blockIdx.y;
    const int chunks = K / (64 * D);
    const int start_chunk = Partitioned ? chunks * partition / splits : 0;
    const int end_chunk = Partitioned ? chunks * (partition + 1) / splits : chunks;
    const int first = start_chunk * D;
    const int last = end_chunk * D;
    const int w_step = N * 64;
    const int w_lane = wave * NBW * 1024 + lane * 16;
    const int w_tile = col * 64;
    const int xfrag = ((lane & 15) * 4) + (lane >> 4);

    const I4 w_resource = resource(weight);
    const I4 x_resource = resource(x);
    I4 wx[D][NBW]{};
    I4 xr[D][XLoads]{};
    I4 xregs[2][MB]{};

    auto load_x = [&](I4 (&buf)[XLoads], int kb) {
        const int k_offset = min(kb, last - 1) * 64;
        unroll<0, XLoads>([&](auto q) {
            constexpr int Q = decltype(q)::value;
            const int row = min(tid / 4 + Q * 64, M - 1);
            load_16<0, false>(buf[Q], x_resource, row * K + (tid & 3) * 16, k_offset);
        });
    };
    auto stage_x = [&](I4 (&buf)[XLoads], int kb) {
        unroll<0, XLoads>([&](auto q) {
            constexpr int Q = decltype(q)::value;
            materialize(buf[Q]);
            lds[(kb & (Slots - 1)) * XSlot + Q * 256 + tid] = buf[Q];
        });
    };
    auto lds_frags = [&](I4 (&buf)[MB], int kb) {
        unroll<0, MB>([&](auto m) {
            constexpr int I = decltype(m)::value;
            buf[I] = lds[(kb & (Slots - 1)) * XSlot + I * 64 + xfrag];
        });
    };
    auto issue = [&](auto d, int kb) {
        constexpr int R = decltype(d)::value;
        unroll<0, NBW>([&](auto n) {
            constexpr int J = decltype(n)::value;
            load_16<J * 1024, NoL2>(wx[R][J], w_resource, w_lane, w_tile + kb * w_step);
        });
        load_x(xr[R], kb + Ahead);
    };

    I4 initial[Ahead][XLoads]{};
    unroll<0, Ahead>([&](auto i) {
        constexpr int A = decltype(i)::value;
        load_x(initial[A], first + A);
    });
    unroll<0, D>([&](auto d) { issue(d, first + decltype(d)::value); });
    wait_vmem<D * Group>();
    unroll<0, Ahead>([&](auto i) {
        constexpr int A = decltype(i)::value;
        stage_x(initial[A], A);
    });
    sync_lds();
    lds_frags(xregs[0], 0);
    sync_lds();

    F4 acc[NBW][MB]{};
    auto consume = [&](auto d, int kb) {
        constexpr int R = decltype(d)::value;
        constexpr int Current = R % 2;
        constexpr int Next = 1 - Current;
        stage_x(xr[R], kb + Ahead);
        lds_frags(xregs[Next], kb + 1);
        unroll<0, NBW>([&](auto n) { materialize(wx[R][decltype(n)::value]); });
        unroll<0, MB>([&](auto m) {
            constexpr int I = decltype(m)::value;
            const I4 a = xregs[Current][I];
            unroll<0, NBW>([&](auto n) {
                constexpr int J = decltype(n)::value;
                const I4 w = wx[R][J];
                const F4 low = mmac(I2{a.x, a.y}, I2{w.x, w.y}, acc[J][I]);
                acc[J][I] = mmac(I2{a.z, a.w}, I2{w.z, w.w}, low);
            });
        });
    };

    for (int base = first; base + D < last; base += D) {
        unroll<0, D>([&](auto d) {
            constexpr int R = decltype(d)::value;
            wait_vmem<(D - 1) * Group>();
            consume(d, base + R - first);
            issue(d, base + D + R);
            if constexpr (R % 2 == 1)
                sync_lds();
        });
    }
    unroll<0, D>([&](auto d) {
        constexpr int R = decltype(d)::value;
        wait_vmem<(D - 1 - R) * Group>();
        consume(d, last - D + R - first);
        if constexpr (R % 2 == 1 && R + 1 < D)
            sync_lds();
    });

    if constexpr (!Partitioned) {
        if (tid < M)
            shared_sx[tid] = sx[tid];
        if (tid < BN)
            shared_sw[tid] = sw[col + tid];
        sync_lds();
    }

    // The offline 32-column permutation places both groups of 4 results
    // adjacent for each lane. One BF16 vector store writes those 8 columns.
    const int lane_col = 8 * (lane >> 4);
    unroll<0, MB>([&](auto m) {
        constexpr int I = decltype(m)::value;
        const int row = I * 16 + (lane & 15);
        if (row >= M)
            return;
        unroll<0, NBW / 2>([&](auto n) {
            constexpr int U = decltype(n)::value;
            const int local_n = waves_n + U * 32 + lane_col;
            const size_t out_index = static_cast<size_t>(row) * N + col + local_n;
            const F4 a = acc[2 * U][I];
            const F4 b = acc[2 * U + 1][I];
            if constexpr (Partitioned) {
                float* dst = partial + (static_cast<size_t>(partition) * M * N) + out_index;
                *reinterpret_cast<F4*>(dst) = a;
                *reinterpret_cast<F4*>(dst + 4) = b;
            } else {
                const F4 sa = *reinterpret_cast<const F4*>(shared_sw + local_n);
                const F4 sb = *reinterpret_cast<const F4*>(shared_sw + local_n + 4);
                const F4 va = (a * shared_sx[row]) * sa;
                const F4 vb = (b * shared_sx[row]) * sb;
                const I4 out{static_cast<int>(bf16_pair(va.x, va.y)), static_cast<int>(bf16_pair(va.z, va.w)),
                             static_cast<int>(bf16_pair(vb.x, vb.y)), static_cast<int>(bf16_pair(vb.z, vb.w))};
                *reinterpret_cast<I4*>(y + out_index) = out;
            }
        });
    });
}

#ifdef EXP4_VEC_REDUCE
template<int FixedSplits>
__global__ void exp4_ring_vec_reduce(const float* __restrict__ scratch,
                                     const float* __restrict__ sx, const float* __restrict__ sw,
                                     uint16_t* __restrict__ y, int M, int N, int splits)
{
    const int col = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (col >= N)
        return;
    const int row = blockIdx.y;
    const size_t plane = static_cast<size_t>(M) * N;
    const size_t i = static_cast<size_t>(row) * N + col;
    F4 a{};
    F4 b{};
    auto accumulate = [&](int s) {
        const float* input = scratch + static_cast<size_t>(s) * plane + i;
        a += *reinterpret_cast<const F4*>(input);
        b += *reinterpret_cast<const F4*>(input + 4);
    };
    if constexpr (FixedSplits != 0) {
        unroll<0, FixedSplits>([&](auto s) { accumulate(decltype(s)::value); });
    } else {
        for (int s = 0; s < splits; ++s)
            accumulate(s);
    }
    const float row_scale = sx[row];
    a = (a * row_scale) * *reinterpret_cast<const F4*>(sw + col);
    b = (b * row_scale) * *reinterpret_cast<const F4*>(sw + col + 4);
    const I4 result{static_cast<int>(bf16_pair(a.x, a.y)),
                    static_cast<int>(bf16_pair(a.z, a.w)),
                    static_cast<int>(bf16_pair(b.x, b.y)),
                    static_cast<int>(bf16_pair(b.z, b.w))};
    *reinterpret_cast<I4*>(y + i) = result;
}
#else
__global__ void exp4_ring_reduce(const float* __restrict__ scratch,
                                 const float* __restrict__ sx, const float* __restrict__ sw,
                                 uint16_t* __restrict__ y, int M, int N, int splits)
{
    const size_t row_n = static_cast<size_t>(M) * N;
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= row_n)
        return;
    float sum = 0;
    for (int s = 0; s < splits; ++s)
        sum += scratch[static_cast<size_t>(s) * row_n + i];
    y[i] = bf16_bits((sum * sx[i / N]) * sw[i % N]);
}
#endif

template<int MB, int D, bool NoL2>
static void go(const uint8_t* x, const uint8_t* w, const float* sx, const float* sw,
               uint16_t* y, float* scratch, int m, int n, int k, int splits, hipStream_t stream)
{
    const dim3 grid(n / TileColumns, splits);
    if (splits > 1) {
        hipLaunchKernelGGL((exp4_ring_gemm<MB, D, true, NoL2>), grid, dim3(256), 0,
                          stream, x, w, sx, sw, y, scratch, m, n, k, splits);
        const auto p = hipPeekAtLastError();
        if (p != hipSuccess)
            return;
#ifdef EXP4_VEC_REDUCE
        constexpr int ReduceThreads = 128;
        constexpr int ReduceColumns = 8 * ReduceThreads;
        const dim3 reduce_grid((n + ReduceColumns - 1) / ReduceColumns, m);
        if (splits == 3)
            hipLaunchKernelGGL((exp4_ring_vec_reduce<3>), reduce_grid, dim3(ReduceThreads), 0,
                              stream, scratch, sx, sw, y, m, n, splits);
        else
            hipLaunchKernelGGL((exp4_ring_vec_reduce<0>), reduce_grid, dim3(ReduceThreads), 0,
                              stream, scratch, sx, sw, y, m, n, splits);
#else
        hipLaunchKernelGGL(exp4_ring_reduce, dim3((m * n + 255) / 256), dim3(256), 0,
                          stream, scratch, sx, sw, y, m, n, splits);
#endif
    } else {
        hipLaunchKernelGGL((exp4_ring_gemm<MB, D, false, NoL2>), grid, dim3(256), 0,
                          stream, x, w, sx, sw, y, scratch, m, n, k, splits);
    }
}

template<int D, bool NoL2>
static void dispatch(const uint8_t* x, const uint8_t* w, const float* sx, const float* sw,
                     uint16_t* y, float* scratch, int m, int n, int k, int splits, hipStream_t stream)
{
    if (m <= 16)
        return go<1, D, NoL2>(x,w,sx,sw,y,scratch,m,n,k,splits,stream);
    if (m <= 32)
        return go<2, D, NoL2>(x,w,sx,sw,y,scratch,m,n,k,splits,stream);
    if (m <= 64)
        return go<4, D, NoL2>(x,w,sx,sw,y,scratch,m,n,k,splits,stream);
    if (m <= 96)
        return go<6, D, NoL2>(x,w,sx,sw,y,scratch,m,n,k,splits,stream);
    return go<8, D, NoL2>(x,w,sx,sw,y,scratch,m,n,k,splits,stream);
}

extern "C" int exp4_ring_launch(
    const uint8_t* x, const uint8_t* w, const float* sx, const float* sw, uint16_t* y,
    float* scratch, int m, int n, int k, int splits, int depth, int cache_w, hipStream_t stream)
{
#ifdef EXP4_VEC_REDUCE
    const bool valid_depth = depth == 2;
#else
    const bool valid_depth = depth == 2 || depth == 4;
#endif
    if (!x || !w || !sx || !sw || !y || m <= 0 || m > 128 || n <= 0 || n % TileColumns ||
        k <= 0 || !valid_depth || k % (64 * depth) || splits < 1 ||
        splits > 8 || k / (64 * depth) < splits || (splits > 1 && !scratch) ||
        static_cast<int64_t>(n) * k >= std::numeric_limits<int32_t>::max()) {
        return hipErrorInvalidValue;
    }
#ifdef EXP4_VEC_REDUCE
    if (cache_w)
        dispatch<2, false>(x,w,sx,sw,y,scratch,m,n,k,splits,stream);
    else
        dispatch<2, true>(x,w,sx,sw,y,scratch,m,n,k,splits,stream);
#else
    if (cache_w && depth == 2)
        dispatch<2, false>(x,w,sx,sw,y,scratch,m,n,k,splits,stream);
    else if (cache_w)
        dispatch<4, false>(x,w,sx,sw,y,scratch,m,n,k,splits,stream);
    else if (depth == 2)
        dispatch<2, true>(x,w,sx,sw,y,scratch,m,n,k,splits,stream);
    else
        dispatch<4, true>(x,w,sx,sw,y,scratch,m,n,k,splits,stream);
#endif
    return hipGetLastError();
}
