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
#ifndef OUTNT
#define OUTNT 0
#endif
template<int O> __device__ __forceinline__ void st(int* p,const i4& r){
 if constexpr(OUTNT)asm volatile("global_store_dwordx4 %0, %1, off offset:%2 glc slc"::"v"(p),"v"(r),"n"(O):"memory");
 else asm volatile("global_store_dwordx4 %0, %1, off offset:%2"::"v"(p),"v"(r),"n"(O):"memory");
}
using u4=__attribute__((ext_vector_type(4))) unsigned;
__device__ __forceinline__ u4 srdesc(const void* p){
 uint64_t a=(uint64_t)p;
 return u4{(unsigned)a,(unsigned)(a>>32)+0x40800000u,0x80000000u,0x00020000u};
}
template<int FM,int FN,int FK>
__global__ __launch_bounds__(NW*64) void reg_kernel(
 const int8_t* __restrict__ xp,const int8_t* __restrict__ wp,
 int* __restrict__ y,int M_arg,int N_arg,int K_arg,int MP_arg) {
 const int M=FM?FM:M_arg,N=FN?FN:N_arg,K=FK?FK:K_arg;
 const int MP=FM?((FM+NW*MB*16-1)/(NW*MB*16)*(NW*MB*16)):MP_arg;
 static_assert(DEPTH==2 && NB%NW==0 && PI,"W shared, X direct, PI output");
 constexpr int BM=NW*MB*16,BN=NB*16,WG=NB/NW;
 __shared__ __align__(16) i4 lwbuf[2*NB*64];
 const int tid=threadIdx.x,lane=tid&63,wave=tid>>6;
 const int kb=K/64,ns=N/16,ms=MP/16;
 const int mbase=(FM?0:blockIdx.y)*BM+wave*MB*16,nbase=blockIdx.x*BN;
 const int8_t* xl=xp+(size_t)(mbase/16)*1024+lane*16;
 const u4 sw=srdesc(wp);
 const unsigned woff=(unsigned)((nbase/16)*1024+tid*16);
 i4 xr[2][MB],acc[MB][NB];
 sf<0,2>([&](auto q){sf<0,MB>([&](auto i){xr[q][i]={0,0,0,0};});});
 sf<0,MB>([&](auto i){sf<0,NB>([&](auto j){acc[i][j]={0,0,0,0};});});
 auto issue=[&](auto sc,int s){
   constexpr int slot=decltype(sc)::value;
   sf<0,MB>([&](auto i){ld<XNT>(xr[slot][i],xl+(size_t)s*ms*1024+i*1024);});
   sf<0,WG>([&](auto q){
     __builtin_amdgcn_raw_buffer_load_lds(sw,
       (__attribute__((address_space(3))) int*)&lwbuf[slot*NB*64+wave*64+q*NW*64],
       16,woff+(unsigned)(s*ns*1024+q*NW*64*16),0,0,0);
   });
 };
 auto body=[&](auto sc,int s){
   constexpr int slot=decltype(sc)::value;
   vw<0>();
   __builtin_amdgcn_s_waitcnt(0xc07f);
   asm volatile("s_barrier":::"memory");
   sf<0,MB>([&](auto i){tie(xr[slot][i]);});
   if(s+1<kb)issue(std::integral_constant<int,1-slot>{},s+1);
   i4 wf[NB];
   unsigned ad=(unsigned)(uintptr_t)&lwbuf[slot*NB*64+lane];
   sf<0,NB>([&](auto j){lr<decltype(j)::value*1024>(wf[decltype(j)::value],ad);});
   sf<0,NB>([&](auto j){
     lw<NB-1-decltype(j)::value>(wf[decltype(j)::value]);
     sf<0,2>([&](auto h){sf<0,MB>([&](auto i){
       acc[i][j]=__builtin_amdgcn_mmac_i32_16x16x32i8(half<h>(xr[slot][i]),half<h>(wf[j]),acc[i][j]);
     });});
   });
 };
 issue(std::integral_constant<int,0>{},0);
 int s=0;
 for(;s+2<=kb;s+=2){body(std::integral_constant<int,0>{},s);body(std::integral_constant<int,1>{},s+1);}
 if(s<kb)body(std::integral_constant<int,0>{},s);
 vw<0>();
 sf<0,2>([&](auto q){sf<0,MB>([&](auto i){tie(xr[q][i]);});});
 sf<0,MB>([&](auto i){
   int m=mbase+i*16+(lane&15),n=nbase+(lane>>4)*4;
   int* yp=y+(size_t)m*N+n;
   if(m<M)sf<0,NB>([&](auto j){st<decltype(j)::value*64>(yp,acc[i][j]);});
 });
 sf<0,MB>([&](auto i){sf<0,NB>([&](auto j){tie(acc[i][j]);});});
}
extern "C" int zmoe_launch(const int8_t* x,const int8_t* w,int* y,int M,int N,int K,void* stream){
 constexpr int BM=NW*MB*16,BN=NB*16;
 if(M<1||N%BN||K<64||K%64)return -1;
 int mp=(M+BM-1)/BM*BM;
 if(M==128&&N==17408&&K==5120)
   reg_kernel<128,17408,5120><<<dim3(N/BN,1),dim3(NW*64),0,(hipStream_t)stream>>>(x,w,y,M,N,K,mp);
 else reg_kernel<0,0,0><<<dim3(N/BN,mp/BM),dim3(NW*64),0,(hipStream_t)stream>>>(x,w,y,M,N,K,mp);
 return (int)hipGetLastError();
}
extern "C" int zmoe_info(int* o){
 hipFuncAttributes a{};auto e=hipFuncGetAttributes(&a,(const void*)reg_kernel<128,17408,5120>);
 o[0]=a.numRegs;o[1]=a.sharedSizeBytes;o[2]=a.localSizeBytes;
 o[3]=NW*MB*16;o[4]=NB*16;o[5]=DEPTH;o[6]=NW;return (int)e;
}

extern "C" int zmoe_info_generic(int* o){
 hipFuncAttributes a{};auto e=hipFuncGetAttributes(&a,(const void*)reg_kernel<0,0,0>);
 o[0]=a.numRegs;o[1]=a.sharedSizeBytes;o[2]=a.localSizeBytes;
 o[3]=NW*MB*16;o[4]=NB*16;o[5]=DEPTH;o[6]=NW;return (int)e;
}
