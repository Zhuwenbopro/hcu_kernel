// TP2 gfx936 implementation based on hip_oneshot_ar/src/hip_oneshot2.cpp.
// Each CTA publishes and consumes the same slice on both ranks. This removes
// the device-wide rendezvous between copy-in and remote reads.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <new>

namespace {
constexpr int max_blocks = 128;
constexpr int threads = 256;
constexpr int hidden_size = 5120;
using U16 = uint16_t;
struct alignas(16) Pack { U16 x[8]; };
struct alignas(128) Signal {
    uint32_t start[max_blocks][2];
    uint32_t done[max_blocks][2];
    uint32_t generation[max_blocks];
};
struct Peers {
    Signal* signal[2];
    Pack* stage[2];
    size_t slot_packs;
};
struct Context {
    Peers peers{};
    int rank;
    int cap;
    int cus;
    int int8_caps[2]{};
    size_t bytes = 0;
};

// Every row of 640 packs has the same owner across plain and fused operations,
// independent of the number of rows. Each active block owns its generation.
// Two slots let the peer finish reading the previous payload after our return.
__device__ __forceinline__ uint32_t next_generation(const Peers& peers, int rank) {
    return 1u + __atomic_load_n(&peers.signal[rank]->generation[blockIdx.x], __ATOMIC_RELAXED);
}

template<bool Dual>
__device__ __forceinline__ void publish(const Peers& peers, int rank, uint32_t gen) {
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        const int b = blockIdx.x;
        const unsigned slot = Dual ? (gen & 1u) : 0u;
        // Separate phase flags prevent a faster peer overwriting a release
        // the slower rank has not observed yet. Finishing this wait certifies
        // that the peer completed all accesses in its previous kernel.
        __atomic_store_n(&peers.signal[1-rank]->start[b][slot], gen, __ATOMIC_RELEASE);
        while (__atomic_load_n(&peers.signal[rank]->start[b][slot], __ATOMIC_ACQUIRE) != gen) {}
        if constexpr (Dual)
            __atomic_store_n(&peers.signal[rank]->generation[b], gen, __ATOMIC_RELAXED);
    }
    __syncthreads();
}

// Legacy C callers may supply only one buffer. Their trailing handshake
// guarantees the peer finished its reads before we reuse that buffer.
__device__ __forceinline__ void finish_single(const Peers& peers, int rank, uint32_t gen) {
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        const int b = blockIdx.x;
        __atomic_store_n(&peers.signal[1-rank]->done[b][rank], gen, __ATOMIC_RELEASE);
        while (__atomic_load_n(&peers.signal[rank]->done[b][1-rank], __ATOMIC_ACQUIRE) != gen) {}
        __atomic_store_n(&peers.signal[rank]->generation[b], gen, __ATOMIC_RELAXED);
    }
    __syncthreads();
}

constexpr int row_packs = hidden_size / 8;
#include "gemma_row.inc"

template<bool Dual>
__global__ __launch_bounds__(threads) void plain_ar(Peers peers, int rank,
                                                     const Pack* inp, Pack* out, int packs) {
    const uint32_t gen = next_generation(peers, rank);
    const size_t offset = Dual ? (gen & 1u) * peers.slot_packs : 0;
    Pack* local = peers.stage[rank] + offset;
    const Pack* remote = peers.stage[1-rank] + offset;
    const int stride = gridDim.x * row_packs;
    for (int row = blockIdx.x * row_packs; row < packs; row += stride) {
        for (int p = threadIdx.x; p < row_packs && row + p < packs; p += blockDim.x)
            local[row+p] = inp[row+p];
    }
    publish<Dual>(peers, rank, gen);
    for (int row = blockIdx.x * row_packs; row < packs; row += stride) {
        for (int p = threadIdx.x; p < row_packs && row + p < packs; p += blockDim.x) {
            const PackedBits own = __builtin_bit_cast(PackedBits, inp[row+p]);
            const PackedBits peer = __builtin_bit_cast(PackedBits, remote[row+p]);
            // Preserve rank order even for competing NaN payloads.
            const PackedBits reduced = rank == 0
                ? allreduce_pairs<true>(own, peer)
                : allreduce_pairs<true>(peer, own);
            out[row+p] = __builtin_bit_cast(Pack, reduced);
        }
    }
    if constexpr (!Dual) finish_single(peers, rank, gen);
}


bool aligned(const void* p) {
    return p && !(reinterpret_cast<uintptr_t>(p) % alignof(Pack));
}

#include "gemma_row_int8.inc"

size_t slot_packs_for(size_t logical_bytes) {
    return (logical_bytes + sizeof(Pack) - 1) / sizeof(Pack);
}

void* allocate_uncached(size_t bytes) {
    void* p = nullptr;
    if (hipExtMallocWithFlags(&p, bytes, hipDeviceMallocUncached) != hipSuccess) return nullptr;
    if (hipMemset(p, 0, bytes) != hipSuccess || hipDeviceSynchronize() != hipSuccess) {
        hipFree(p);
        return nullptr;
    }
    return p;
}
template<bool Dual>
int resident_block_limit() {
    int fused, plain;
    if (hipOccupancyMaxActiveBlocksPerMultiprocessor(&fused, fused_ar<Dual>, norm_threads, 0) != hipSuccess ||
        hipOccupancyMaxActiveBlocksPerMultiprocessor(&plain, plain_ar<Dual>, threads, 0) != hipSuccess) return 0;
    return std::min(fused, plain);
}

int set_staging(void* ctx, void* self, void* const* bufs, int bytes, bool dual) {
    Context* c = static_cast<Context*>(ctx);
    if (!c || !bufs || c->bytes || bytes < 16 || !aligned(self) || self != bufs[c->rank] ||
        !aligned(bufs[1-c->rank])) return -1;
    const size_t slot_packs = dual ? slot_packs_for(bytes) : 0;
    const size_t needed = dual ? 2 * slot_packs * sizeof(Pack) : size_t(bytes);
    size_t allocated = 0;
    if (hipMemPtrGetInfo(self, &allocated) != hipSuccess || needed > allocated) return -1;
    for (int i = 0; i < 2; ++i) c->peers.stage[i] = static_cast<Pack*>(bufs[i]);
    c->peers.slot_packs = slot_packs;
    c->bytes = bytes;
    if (dual)
        set_int8_block_limits<true>(c);
    else
        set_int8_block_limits<false>(c);
    return 0;
}
}  // namespace

extern "C" {
void* osar_meta_alloc(int bytes, int uncached) {
    return uncached && bytes > 0 ? allocate_uncached(bytes) : nullptr;
}
void* osar_staging_alloc(int logical_bytes) {
    if (logical_bytes < 16) return nullptr;
    return allocate_uncached(2 * slot_packs_for(logical_bytes) * sizeof(Pack));
}
void osar_meta_free(void* p) { if (p) hipFree(p); }
int osar_get_handle(void* p, char* out, int n) {
    hipIpcMemHandle_t h;
    if (!p || !out || n < static_cast<int>(sizeof(h))) return -1;
    if (hipIpcGetMemHandle(&h, p) != hipSuccess) return -1;
    memcpy(out, &h, sizeof(h));
    return 0;
}
void* osar_open(const char* buf) {
    if (!buf) return nullptr;
    hipIpcMemHandle_t h;
    memcpy(&h, buf, sizeof(h));
    void* p = nullptr;
    return hipIpcOpenMemHandle(&p, h, hipIpcMemLazyEnablePeerAccess) == hipSuccess ? p : nullptr;
}
int osar_close_handle(void* p) {
    return static_cast<int>(hipIpcCloseMemHandle(p));
}
void* osar_init(int rank, int ws, void* self, void* const* metas, int blocks) {
    if (ws != 2 || rank < 0 || rank >= ws || !self || !metas || self != metas[rank] ||
        !metas[1-rank] || blocks < 1 || blocks > max_blocks) return nullptr;
    int device, cus;
    if (hipGetDevice(&device) != hipSuccess ||
        hipDeviceGetAttribute(&cus, hipDeviceAttributeMultiprocessorCount, device) != hipSuccess ||
        blocks > cus * std::min(resident_block_limit<true>(), resident_block_limit<false>())) return nullptr;
    Context* c = new(std::nothrow) Context();
    if (!c) return nullptr;
    c->rank = rank;
    c->cap = blocks;
    c->cus = cus;
    for (int i = 0; i != 2; ++i) c->peers.signal[i] = static_cast<Signal*>(metas[i]);
    return c;
}
int osar_set_regbuf_sized(void* ctx, void* self, void* const* bufs, int bytes) {
    return set_staging(ctx, self, bufs, bytes, false);
}
int osar_set_regbuf_dual(void* ctx, void* self, void* const* bufs, int logical_bytes) {
    return set_staging(ctx, self, bufs, logical_bytes, true);
}
int osar_set_regbuf(void* ctx, void* self, void* const* bufs) {
    size_t bytes = 0;
    if (hipMemPtrGetInfo(self, &bytes) != hipSuccess || bytes > INT32_MAX) return -1;
    return osar_set_regbuf_sized(ctx, self, bufs, static_cast<int>(bytes));
}
int osar_run_ci(void* ctx, void* inp, void* out, int bytes, int variant, void* stream) {
    Context* c = static_cast<Context*>(ctx);
    if (!c || variant != 0 || bytes <= 0 || bytes % 16 ||
        static_cast<size_t>(bytes) > c->bytes || !aligned(inp) || !aligned(out)) return -1;
    int packs = bytes / 16, blocks = std::min(c->cap, (packs + row_packs-1)/row_packs);
    void* args[] = {&c->peers, &c->rank, &inp, &out, &packs};
    return static_cast<int>(hipLaunchKernel(reinterpret_cast<const void*>(c->peers.slot_packs ? plain_ar<true> : plain_ar<false>),
        dim3(blocks), dim3(threads), args, 0, static_cast<hipStream_t>(stream)));
}
int osar_run_fused(void* ctx, void* inp, void* resid, void* w, void* out_norm,
                   void* out_resid, int rows, int cols, int copyin, float eps,
                   void* stream) {
    Context* c = static_cast<Context*>(ctx);
    int bf16_weight = (copyin == 2);
    if (!c || (copyin != 1 && copyin != 2) || rows < 1 || rows > 4096 || cols != hidden_size ||
        !aligned(inp) || !aligned(resid) || !aligned(w) || !aligned(out_norm) ||
        !aligned(out_resid) || out_norm == out_resid ||
        rows * cols * sizeof(U16) > c->bytes) return -1;
    void* args[] = {&c->peers, &c->rank, &inp, &resid, &w, &out_norm, &out_resid, &rows, &eps, &bf16_weight};
    return static_cast<int>(hipLaunchKernel(reinterpret_cast<const void*>(c->peers.slot_packs ? fused_ar<true> : fused_ar<false>),
        dim3(std::min(c->cap, rows)), dim3(norm_threads), args, 0, static_cast<hipStream_t>(stream)));
}
int osar_run_fused_int8(void* ctx, void* input, void* residual, void* weight, void* q,
                       float* scales, void* residual_output, int rows, int cols, int copyin,
                       float epsilon, void* stream, int selected_cap, int quant_mode) {
    return run_fused_int8<false>(ctx, input, residual, weight, q, scales, residual_output, rows,
                                cols, copyin, epsilon, stream, selected_cap, quant_mode);
}
int osar_fused_int8_block_limit(void* ctx, int mode) {
    const auto* c = static_cast<const Context*>(ctx);
    if (!c || (mode != 0 && mode != 1))
        return 0;
    return c->int8_caps[mode];
}
int osar_dev_synchronize() { return static_cast<int>(hipDeviceSynchronize()); }
int osar_dispose(void* ctx) {
    Context* c = static_cast<Context*>(ctx);
    if (!c) return 0;
    int peer = 1-c->rank;
    hipError_t data_rc = c->peers.stage[peer] ? hipIpcCloseMemHandle(c->peers.stage[peer]) : hipSuccess;
    hipError_t flag_rc = hipIpcCloseMemHandle(c->peers.signal[peer]);
    delete c;
    return static_cast<int>(data_rc != hipSuccess ? data_rc : flag_rc);
}
}
