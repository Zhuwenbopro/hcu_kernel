#include <hip/hip_runtime.h>
#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstring>
#include <new>

namespace {
constexpr int ranks = 4;
constexpr int max_blocks = 80;
constexpr int hidden_size = 5120;
constexpr int row_packs = hidden_size / 8;
constexpr int shard_packs = row_packs / ranks;
struct alignas(16) Pack { uint16_t x[8]; };
using Counter = unsigned long long;
struct alignas(128) Signal {
    Counter ready[max_blocks][2][ranks];
    Counter reduced[max_blocks][2][ranks];
    Counter generation[max_blocks];
};
struct Peers {
    Signal* signal[ranks];
    Pack* stage[ranks];
    size_t full_packs;
};
struct Context { Peers peers; int rank; int cap; };

#include "gemma_exact.inc"

__device__ __forceinline__ PackedBits reduce_four(PackedBits a, PackedBits b,
                                                  PackedBits c, PackedBits d) {
    FloatPair sums[4], aa[4], bb[4], cc[4], dd[4];
#pragma unroll
    for (int j=0; j<4; ++j) {
        aa[j]=unpack_bf16(a[j]); bb[j]=unpack_bf16(b[j]);
        cc[j]=unpack_bf16(c[j]); dd[j]=unpack_bf16(d[j]);
    }
    asm volatile(
        "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0\n\t"
        "s_nop 1\n\t"
        "v_pk_add_f32 %0, %4, %8\n\t"
        "v_pk_add_f32 %1, %5, %9\n\t"
        "v_pk_add_f32 %2, %6, %10\n\t"
        "v_pk_add_f32 %3, %7, %11\n\t"
        "v_pk_add_f32 %0, %0, %12\n\t"
        "v_pk_add_f32 %1, %1, %13\n\t"
        "v_pk_add_f32 %2, %2, %14\n\t"
        "v_pk_add_f32 %3, %3, %15\n\t"
        "v_pk_add_f32 %0, %0, %16\n\t"
        "v_pk_add_f32 %1, %1, %17\n\t"
        "v_pk_add_f32 %2, %2, %18\n\t"
        "v_pk_add_f32 %3, %3, %19\n\t"
        "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3\n\t"
        "s_nop 1"
        : "=&v"(sums[0]), "=&v"(sums[1]), "=&v"(sums[2]), "=&v"(sums[3])
        : "v"(aa[0]),"v"(aa[1]),"v"(aa[2]),"v"(aa[3]),
          "v"(bb[0]),"v"(bb[1]),"v"(bb[2]),"v"(bb[3]),
          "v"(cc[0]),"v"(cc[1]),"v"(cc[2]),"v"(cc[3]),
          "v"(dd[0]),"v"(dd[1]),"v"(dd[2]),"v"(dd[3]));
    PackedBits result;
#pragma unroll
    for(int j=0;j<4;++j) result[j] = pack_bf16(sums[j]);
    return result;
}

// The same CTA owns each row's input, temporary shard and generation at all M.
// Each release covers every lane's writes, then the rank waits for three peers.
// Generation parity prevents clobbering a peer that is still finishing its reads.
template<bool ShardReady>
__device__ __forceinline__ void rendezvous(Peers peers, int rank, Counter gen) {
    __threadfence_system();
    __syncthreads();
    int peer=threadIdx.x, b=blockIdx.x, slot=int(gen&1);
    if(peer<ranks && peer!=rank) {
        auto* send = ShardReady ? &peers.signal[peer]->reduced[b][slot][rank]
                                : &peers.signal[peer]->ready[b][slot][rank];
        auto* recv = ShardReady ? &peers.signal[rank]->reduced[b][slot][peer]
                                : &peers.signal[rank]->ready[b][slot][peer];
        __atomic_store_n(send, gen, __ATOMIC_RELEASE);
        while(__atomic_load_n(recv,__ATOMIC_ACQUIRE)!=gen) {}
    }
    __syncthreads();
}

__device__ __forceinline__ PackedBits read_rank(Peers peers, int rank, int source,
    size_t offset, int idx, PackedBits own) {
    return source == rank ? own : load_pack(peers.stage[source] + offset + idx);
}

__device__ __forceinline__ PackedBits ordered_sum(Peers peers, int rank,
    size_t offset, int idx, int rows, PackedBits own) {
    // Match AITER one-stage order below 160 KiB and two-stage cyclic owner
    // order thereafter. Its partitions are contiguous over the *flat* tensor.
    int part=rows*shard_packs;
    int first= rows<16 ? 0 : (idx>=part)+(idx>=2*part)+(idx>=3*part);
    return reduce_four(read_rank(peers,rank,first,offset,idx,own),
                       read_rank(peers,rank,(first+1)&3,offset,idx,own),
                       read_rank(peers,rank,(first+2)&3,offset,idx,own),
                       read_rank(peers,rank,(first+3)&3,offset,idx,own));
}

template<bool Shard>
__global__ __launch_bounds__(norm_threads) void fused(Peers peers, int rank,
    const Pack* input,const Pack* residual,const Pack* weight,Pack* output,
    Pack* residual_output,int rows,float epsilon) {
    const int b=blockIdx.x, pack=threadIdx.x, stride=gridDim.x;
    __shared__ Counter shared_gen;
    if(pack==0) {
        shared_gen=1+__atomic_load_n(&peers.signal[rank]->generation[b],__ATOMIC_RELAXED);
        __atomic_store_n(&peers.signal[rank]->generation[b],shared_gen,__ATOMIC_RELAXED);
    }
    __syncthreads();
    Counter gen=shared_gen;
    size_t slot = gen&1;
    size_t offset=slot*peers.full_packs;
    size_t scratch_offset=2*peers.full_packs+slot*(peers.full_packs/4);
    Pack* local=peers.stage[rank]+offset;
    int initial_idx=b*row_packs+pack;
    PackedBits initial=load_pack(input+initial_idx);
    local[initial_idx]=__builtin_bit_cast(Pack,initial);
    for(int row=b+stride;row<rows;row+=stride) {
        int idx=row*row_packs+pack;
        local[idx]=input[idx];
    }
    rendezvous<false>(peers,rank,gen);
    if constexpr(Shard) {
        // Divide every row into four consecutive quarters. A single CTA on
        // each rank owns all quarters of a row across both communication phases.
        int intra=pack%shard_packs;
        for(int row=b+(pack/shard_packs)*stride;row<rows;row+=4*stride) {
            int idx=row*row_packs+rank*shard_packs+intra;
            PackedBits reduced=ordered_sum(peers,rank,offset,idx,rows,load_pack(input+idx));
            peers.stage[rank][scratch_offset+row*shard_packs+intra]=__builtin_bit_cast(Pack,reduced);
        }
        rendezvous<true>(peers,rank,gen);
    }
    __shared__ float wave_sums[row_waves];
    for(int row=b;row<rows;row+=stride) {
        int idx=row*row_packs+pack;
        PackedBits reduced;
        if constexpr(Shard) {
            reduced=load_pack(peers.stage[pack/shard_packs]+scratch_offset+
                              row*shard_packs+pack%shard_packs);
        } else {
            PackedBits own=row==b ? initial : load_pack(input+idx);
            reduced=ordered_sum(peers,rank,offset,idx,rows,own);
        }
        finish_norm(reduced,residual,weight,output,residual_output,idx,wave_sums,epsilon);
        if(row+stride<rows) __syncthreads();
    }
}

bool aligned(const void* p) {return p && reinterpret_cast<uintptr_t>(p)%16==0;}
bool valid_bytes(size_t b) {return b>=hidden_size*2 && b%(hidden_size*2)==0 && b/16 <= INT_MAX;}
}
extern "C" {
size_t tp4_signal_bytes(int blocks) {return blocks>0 && blocks<=max_blocks?sizeof(Signal):0;}
size_t tp4_data_bytes(size_t b) {return valid_bytes(b)?2*(b+b/4):0;}
void* tp4_alloc(size_t b) {
    void* p=nullptr;
    if(!b||hipExtMallocWithFlags(&p,b,hipDeviceMallocUncached)!=hipSuccess) return nullptr;
    if(hipMemset(p,0,b)!=hipSuccess || hipDeviceSynchronize()!=hipSuccess) {hipFree(p);return nullptr;}
    return p;
}
int tp4_free(void* p) {return int(hipFree(p));}
int tp4_handle(void* p,void* dst) {
    if(!p||!dst) return -1;
    hipIpcMemHandle_t h;
    static_assert(sizeof(h)==64);
    auto err=hipIpcGetMemHandle(&h,p);
    if(err==hipSuccess) std::memcpy(dst,&h,sizeof(h));
    return int(err);
}
void* tp4_open(void* src) {
    if(!src) return nullptr;
    hipIpcMemHandle_t h;
    std::memcpy(&h,src,sizeof(h));
    void* p=nullptr;
    return hipIpcOpenMemHandle(&p,h,hipIpcMemLazyEnablePeerAccess)==hipSuccess?p:nullptr;
}
int tp4_close(void* p) {return int(hipIpcCloseMemHandle(p));}
void* tp4_create(int rank,void** signals,void** data,size_t maximum_bytes,int blocks) {
    if(rank<0||rank>=ranks||!signals||!data||!tp4_signal_bytes(blocks)||!valid_bytes(maximum_bytes)) return nullptr;
    size_t size=0;
    for(int i=0;i<ranks;++i) if(!aligned(signals[i])||!aligned(data[i])) return nullptr;
    if(hipMemPtrGetInfo(signals[rank],&size)!=hipSuccess || size<sizeof(Signal)) return nullptr;
    if(hipMemPtrGetInfo(data[rank],&size)!=hipSuccess || size<tp4_data_bytes(maximum_bytes)) return nullptr;
    int direct=0,shard=0,device=0,cus=0;
    if(hipOccupancyMaxActiveBlocksPerMultiprocessor(&direct,fused<false>,norm_threads,0)!=hipSuccess ||
       hipOccupancyMaxActiveBlocksPerMultiprocessor(&shard,fused<true>,norm_threads,0)!=hipSuccess ||
       hipGetDevice(&device)!=hipSuccess ||
       hipDeviceGetAttribute(&cus,hipDeviceAttributeMultiprocessorCount,device)!=hipSuccess ||
       blocks>std::min(direct,shard)*cus) return nullptr;
    Context* ctx=new(std::nothrow) Context{};
    if(!ctx) return nullptr;
    ctx->rank=rank; ctx->cap=blocks; ctx->peers.full_packs=maximum_bytes/sizeof(Pack);
    for(int i=0;i<ranks;++i){ctx->peers.signal[i]=static_cast<Signal*>(signals[i]);ctx->peers.stage[i]=static_cast<Pack*>(data[i]);}
    return ctx;
}
int tp4_destroy(void* ctx){delete static_cast<Context*>(ctx);return 0;}
int tp4_run(void* opaque,int alg,void* x,void* r,void* w,void* y,void* ro,int rows,float eps,void* stream) {
    auto* c=static_cast<Context*>(opaque);
    if(!c || (alg!=0 && alg!=1) || rows<=0 || size_t(rows)*row_packs>c->peers.full_packs ||
       !aligned(x)||!aligned(r)||!aligned(w)||!aligned(y)||!aligned(ro)) return -1;
    int grid=std::min(rows,c->cap);
    auto launch=[&](auto k) {hipLaunchKernelGGL(k,dim3(grid),dim3(norm_threads),0,
        static_cast<hipStream_t>(stream),c->peers,c->rank,static_cast<const Pack*>(x),
        static_cast<const Pack*>(r),static_cast<const Pack*>(w),static_cast<Pack*>(y),
        static_cast<Pack*>(ro),rows,eps);};
    if(alg==0) launch(fused<false>); else launch(fused<true>);
    return int(hipGetLastError());
}
}
