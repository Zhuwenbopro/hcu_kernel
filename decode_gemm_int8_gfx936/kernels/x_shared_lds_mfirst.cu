#include <hip/hip_runtime.h>
#include <stdint.h>
#include <type_traits>
#ifndef MB
#define MB 2
#endif
#ifndef NB
#define NB 2
#endif
#ifndef NW
#define NW 4
#endif
#ifndef DEPTH
#define DEPTH 1
#endif
#ifndef WNT
#define WNT 1
#endif
#ifndef XNT
#define XNT 0
#endif
#ifndef NOMMA
#define NOMMA 0
#endif
using i4=__attribute__((ext_vector_type(4))) int;
using i2=__attribute__((ext_vector_type(2))) int;
template<int I,int N,class F> __device__ __forceinline__ void sf(F&& f) {
 if constexpr(I<N){f(std::integral_constant<int,I>{});sf<I+1,N>(f);}
}
template<int NT> __device__ __forceinline__ void ld(i4& r,const int8_t* p) {
 if constexpr(NT) asm volatile("global_load_dwordx4 %0, %1, off glc slc" : "+v"(r):"v"(p));
 else asm volatile("global_load_dwordx4 %0, %1, off" : "+v"(r):"v"(p));
}
__device__ __forceinline__ void tie(i4& r){asm volatile("" : "+v"(r));}
template<int N> __device__ __forceinline__ void vw(){asm volatile("s_waitcnt vmcnt(%0)"::"n"(N));}
template<int H> __device__ __forceinline__ long half(const i4& r){
 if constexpr(H==0)return __builtin_bit_cast(long,i2{r.x,r.y});
 else return __builtin_bit_cast(long,i2{r.z,r.w});
}
template<int O> __device__ __forceinline__ void lr(i4& r,unsigned ad){asm volatile("ds_read_b128 %0, %1 offset:%2":"=v"(r):"v"(ad),"n"(O));}
template<int N> __device__ __forceinline__ void lw(i4& r){asm volatile("s_waitcnt lgkmcnt(%1)": "+v"(r):"n"(N):"memory");}
using u4=__attribute__((ext_vector_type(4))) unsigned;
__device__ __forceinline__ u4 srdesc(const void* p){
 uint64_t a=(uint64_t)p;
 return u4{(unsigned)a,(unsigned)(a>>32)+0x40800000u,0x80000000u,0x00020000u};
}
__global__ __launch_bounds__(NW*64) void reg_kernel(
 const int8_t* __restrict__ xp,const int8_t* __restrict__ wp,
 int* __restrict__ y,int M,int N,int K,int MP) {
 static_assert(DEPTH==2 && MB%NW==0,"one-stage future prefetch");
 constexpr int BN=NW*NB*16,XG=MB/NW;
 __shared__ __align__(16) i4 lx[2*MB*64];
 const int tid=threadIdx.x,lane=tid&63,wave=tid>>6;
 const int kb=K/64,ns=N/16,ms=MP/16;
 const int mbase=blockIdx.x*(MB*16),nbase=blockIdx.y*BN+wave*NB*16;
 const int8_t* wl=wp+(size_t)(nbase/16)*1024+lane*16;
 const u4 sx=srdesc(xp);
 const unsigned xoff=(unsigned)((mbase/16)*1024+tid*16);
 i4 wr[2][NB],acc[MB][NB];
 sf<0,2>([&](auto q){sf<0,NB>([&](auto j){wr[q][j]={0,0,0,0};});});
 sf<0,MB>([&](auto i){sf<0,NB>([&](auto j){acc[i][j]={0,0,0,0};});});
 auto issue=[&](auto sc,int s){
   constexpr int slot=decltype(sc)::value;
   sf<0,NB>([&](auto j){ld<WNT>(wr[slot][j],wl+(size_t)s*ns*1024+j*1024);});
   sf<0,XG>([&](auto q){
     __builtin_amdgcn_raw_buffer_load_lds(sx,
       (__attribute__((address_space(3))) int*)&lx[slot*MB*64+wave*64+q*NW*64],
       16,xoff+(unsigned)(s*ms*1024+q*NW*64*16),0,0,0);
   });
 };
 auto body=[&](auto sc,int s){
   constexpr int slot=decltype(sc)::value;
   vw<0>();
   __builtin_amdgcn_s_waitcnt(0xc07f);
   asm volatile("s_barrier":::"memory");
   sf<0,NB>([&](auto j){tie(wr[slot][j]);});
   if(s+1<kb)issue(std::integral_constant<int,1-slot>{},s+1);
   i4 xf[MB];
   unsigned ad=(unsigned)(uintptr_t)&lx[slot*MB*64+lane];
   sf<0,MB>([&](auto i){
     lr<decltype(i)::value*1024>(xf[decltype(i)::value],ad);
   });
   sf<0,MB>([&](auto i){
     lw<MB-1-decltype(i)::value>(xf[decltype(i)::value]);
     sf<0,2>([&](auto h){sf<0,NB>([&](auto j){
       acc[i][j]=__builtin_amdgcn_mmac_i32_16x16x32i8(half<h>(xf[i]),half<h>(wr[slot][j]),acc[i][j]);
     });});
   });
 };
 issue(std::integral_constant<int,0>{},0);
 int s=0;
 for(;s+2<=kb;s+=2){body(std::integral_constant<int,0>{},s);body(std::integral_constant<int,1>{},s+1);}
 if(s<kb)body(std::integral_constant<int,0>{},s);
 vw<0>();
 sf<0,2>([&](auto q){sf<0,NB>([&](auto j){tie(wr[q][j]);});});
#if PI
 sf<0,MB>([&](auto i){sf<0,NB>([&](auto j){
   int m=mbase+i*16+(lane&15),n=nbase+j*16+(lane>>4)*4;
   if(m<M)*reinterpret_cast<i4*>(y+(size_t)m*N+n)=acc[i][j];
 });});
#else
 sf<0,MB>([&](auto i){sf<0,NB>([&](auto j){sf<0,4>([&](auto e){
   int m=mbase+i*16+(lane&15),n=nbase+j*16+(lane>>4)+4*e;
   if(m<M)y[(size_t)m*N+n]=acc[i][j][decltype(e)::value];
 });});});
#endif
}
extern "C" int zmoe_launch(const int8_t* x,const int8_t* w,int* y,int M,int N,int K,void* stream){
 constexpr int BM=MB*16,BN=NW*NB*16;
 if(M<1||N%BN||K<64||K%64)return -1;
 int mp=(M+BM-1)/BM*BM;
 reg_kernel<<<dim3(mp/BM,N/BN),dim3(NW*64),0,(hipStream_t)stream>>>(x,w,y,M,N,K,mp);
 return (int)hipGetLastError();
}
extern "C" int zmoe_info(int* o){
 hipFuncAttributes a{};auto e=hipFuncGetAttributes(&a,(const void*)reg_kernel);
 o[0]=a.numRegs;o[1]=a.sharedSizeBytes;o[2]=a.localSizeBytes;
 o[3]=MB*16;o[4]=NW*NB*16;o[5]=DEPTH;o[6]=NW;return (int)e;
}
