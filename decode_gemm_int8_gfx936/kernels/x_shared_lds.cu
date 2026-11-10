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
using u4=__attribute__((ext_vector_type(4))) unsigned;
__device__ __forceinline__ u4 srdesc(const void* p){
 uint64_t a=(uint64_t)p;
 return u4{(unsigned)a,(unsigned)(a>>32)+0x40800000u,0x80000000u,0x00020000u};
}
__global__ __launch_bounds__(NW*64) void reg_kernel(
 const int8_t* __restrict__ xp,const int8_t* __restrict__ wp,
 int* __restrict__ y,int M,int N,int K,int MP) {
 static_assert(DEPTH==1 && MB%NW==0,"E009 uses a serial shared-X delivery baseline");
 constexpr int BN=NW*NB*16,XG=MB/NW;
 __shared__ __align__(16) i4 lx[2*MB*64];
 const int tid=threadIdx.x,lane=tid&63,wave=tid>>6;
 const int kb=K/64,ns=N/16,ms=MP/16;
 const int mbase=blockIdx.y*(MB*16),nbase=blockIdx.x*BN+wave*NB*16;
 const int8_t* wl=wp+(size_t)(nbase/16)*1024+lane*16;
 const u4 sx=srdesc(xp);
 const unsigned xoff=(unsigned)((mbase/16)*1024+tid*16);
 i4 acc[MB][NB];
 sf<0,MB>([&](auto i){sf<0,NB>([&](auto j){acc[i][j]={0,0,0,0};});});
 for(int s=0;s<kb;++s){
   i4 wr[NB],xf[MB];
   sf<0,NB>([&](auto j){wr[j]={0,0,0,0};ld<WNT>(wr[j],wl+(size_t)s*ns*1024+j*1024);});
   sf<0,XG>([&](auto q){
     __builtin_amdgcn_raw_buffer_load_lds(sx,
       (__attribute__((address_space(3))) int*)&lx[(s&1)*MB*64+wave*64+q*NW*64],
       16,xoff+(unsigned)(s*ms*1024+q*NW*64*16),0,0,0);
   });
   vw<0>();
   sf<0,NB>([&](auto j){tie(wr[j]);});

   __builtin_amdgcn_s_waitcnt(0xc07f);
   asm volatile("s_barrier":::"memory");
   sf<0,MB>([&](auto i){xf[i]=lx[(s&1)*MB*64+i*64+lane];});
   __builtin_amdgcn_s_waitcnt(0xc07f);
   sf<0,2>([&](auto h){sf<0,MB>([&](auto i){sf<0,NB>([&](auto j){
#if !NOMMA
     acc[i][j]=__builtin_amdgcn_mmac_i32_16x16x32i8(half<h>(xf[i]),half<h>(wr[j]),acc[i][j]);
#endif
   });});});
 }
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
 reg_kernel<<<dim3(N/BN,mp/BM),dim3(NW*64),0,(hipStream_t)stream>>>(x,w,y,M,N,K,mp);
 return (int)hipGetLastError();
}
extern "C" int zmoe_info(int* o){
 hipFuncAttributes a{};auto e=hipFuncGetAttributes(&a,(const void*)reg_kernel);
 o[0]=a.numRegs;o[1]=a.sharedSizeBytes;o[2]=a.localSizeBytes;
 o[3]=MB*16;o[4]=NW*NB*16;o[5]=DEPTH;o[6]=NW;return (int)e;
}
