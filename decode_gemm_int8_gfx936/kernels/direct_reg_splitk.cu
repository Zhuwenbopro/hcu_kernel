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
#ifndef KS
#define KS 2
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
__global__ __launch_bounds__(NW*64) void reg_kernel(
 const int8_t* __restrict__ xp,const int8_t* __restrict__ wp,
 int* __restrict__ y,int M,int N,int K,int MP) {
 constexpr int BN=(NW/KS)*NB*16,L=NB+MB,D=DEPTH;
 const int lane=threadIdx.x&63,wave=(threadIdx.x>>6)/KS,kwave=(threadIdx.x>>6)%KS;
 __shared__ i4 red[(KS-1)*(NW/KS)*MB*NB*64];
 const int global_kb=K/64,part=(global_kb+KS-1)/KS,kb0=kwave*part;
 const int kb=max(0,min(part,global_kb-kb0)),ns=N/16,ms=MP/16;
 const int mbase=blockIdx.y*(MB*16),nbase=blockIdx.x*BN+wave*NB*16;
 const int8_t* wl=wp+(size_t)(nbase/16)*1024+lane*16;
 const int8_t* xl=xp+(size_t)(mbase/16)*1024+lane*16;
 i4 wr[D][NB],xr[D][MB],acc[MB][NB];
 sf<0,D>([&](auto d){sf<0,NB>([&](auto j){wr[d][j]={0,0,0,0};});
                         sf<0,MB>([&](auto i){xr[d][i]={0,0,0,0};});});
 sf<0,MB>([&](auto i){sf<0,NB>([&](auto j){acc[i][j]={0,0,0,0};});});
 int issued=0,consumed=0;
 auto issue=[&](auto d){
   int s=min(kb0+min(issued,max(kb-1,0)),global_kb-1);
   sf<0,NB>([&](auto j){ld<WNT>(wr[d][j],wl+(size_t)s*ns*1024+j*1024);});
   sf<0,MB>([&](auto i){ld<XNT>(xr[d][i],xl+(size_t)s*ms*1024+i*1024);});
   ++issued;
 };
 auto consume=[&](auto d,auto check){
   sf<0,NB>([&](auto j){tie(wr[d][j]);});
   sf<0,MB>([&](auto i){tie(xr[d][i]);});
   if(!decltype(check)::value || consumed<kb){
     sf<0,2>([&](auto h){sf<0,MB>([&](auto i){sf<0,NB>([&](auto j){
#if !NOMMA
       acc[i][j]=__builtin_amdgcn_mmac_i32_16x16x32i8(half<h>(xr[d][i]),half<h>(wr[d][j]),acc[i][j]);
#endif
     });});});
   }
   ++consumed;
 };
 sf<0,D>([&](auto d){issue(d);});
 int sr=(kb+D-1)/D*D;
 for(int s=D;s<sr;s+=D){
   sf<0,D>([&](auto d){vw<(D-1)*L>();consume(d,std::false_type{});issue(d);});
 }
 sf<0,D>([&](auto d){vw<(D-1-decltype(d)::value)*L>();consume(d,std::true_type{});});
 vw<0>();
 sf<0,D>([&](auto d){sf<0,NB>([&](auto j){tie(wr[d][j]);});
                         sf<0,MB>([&](auto i){tie(xr[d][i]);});});
 if(kwave){
   sf<0,MB>([&](auto i){sf<0,NB>([&](auto j){
     red[((kwave-1)*(NW/KS)*MB*NB+wave*MB*NB+i*NB+j)*64+lane]=acc[i][j];
   });});
 }
 __syncthreads();
 if(kwave)return;
 sf<1,KS>([&](auto q){sf<0,MB>([&](auto i){sf<0,NB>([&](auto j){
   acc[i][j]+=red[((q-1)*(NW/KS)*MB*NB+wave*MB*NB+i*NB+j)*64+lane];
 });});});
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
 constexpr int BM=MB*16,BN=(NW/KS)*NB*16;
 if(M<1||N%BN||K<64||K%64)return -1;
 int mp=(M+BM-1)/BM*BM;
 reg_kernel<<<dim3(N/BN,mp/BM),dim3(NW*64),0,(hipStream_t)stream>>>(x,w,y,M,N,K,mp);
 return (int)hipGetLastError();
}
extern "C" int zmoe_info(int* o){
 hipFuncAttributes a{};auto e=hipFuncGetAttributes(&a,(const void*)reg_kernel);
 o[0]=a.numRegs;o[1]=a.sharedSizeBytes;o[2]=a.localSizeBytes;
 o[3]=MB*16;o[4]=(NW/KS)*NB*16;o[5]=DEPTH;o[6]=NW;return (int)e;
}
