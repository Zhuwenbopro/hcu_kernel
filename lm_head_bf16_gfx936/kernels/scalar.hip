#include <hip/hip_runtime.h>
#include <type_traits>
#include <stdint.h>
using u16=uint16_t;
using u32=uint32_t;
using u2=u32 __attribute__((ext_vector_type(2)));
using u4=u32 __attribute__((ext_vector_type(4)));
constexpr int K=5120,N=124160;
__device__ inline u16 rne(float a){u32 v=__float_as_uint(a); if((v&0x7fffffff)>0x7f800000) return (v>>16)|0x40;return (v+0x7fff+((v>>16)&1))>>16;}
template<int M,int V,int S,int P>
__global__ __launch_bounds__(S*64) void vector_kn(const u16* __restrict__ x,const u16* __restrict__ w,u16* __restrict__ y){
  int lane=threadIdx.x%64,wave=threadIdx.x/64; int col=(blockIdx.x*64+lane)*V;
  bool valid=col<N;
  int k0=0;
  float a[M][V]={};
  for(int k=0;k<K/S;k+=P){
    using WT=std::conditional_t<V==8,u4,u2>;
    WT weight[P];
    u32 xb[P][M];
    int wave_id=__builtin_amdgcn_readfirstlane(wave);
    #pragma unroll
    for(int s=0;s<P;s++){
      #pragma unroll
      for(int m=0;m<M;m++){
        auto ptr=reinterpret_cast<const u32*>(x+m*K)+((k+s)*S+wave_id)/2;
        asm volatile("s_load_dword %0, %1, 0x0" : "=s"(xb[s][m]):"s"(ptr):"memory");
      }
    }

    #pragma unroll
    for(int j=0;j<P;j++){
      const u16* ptr=w+((k+j)*S+wave)*N+col;
      if(!valid) weight[j]={};
      else if constexpr(V==8) asm volatile("global_load_dwordx4 %0, %1, off glc slc" : "=v"(weight[j]):"v"(ptr):"memory");
      else asm volatile("global_load_dwordx2 %0, %1, off glc slc" : "=v"(weight[j]):"v"(ptr):"memory");
    }
    asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
    #pragma unroll
    for(int j=0;j<P;j++){
      #pragma unroll
      for(int m=0;m<M;m++){
        float xv=__uint_as_float(wave_id%2?(xb[j][m]&0xffff0000u):(xb[j][m]<<16));
        #pragma unroll
        for(int b=0;b<V;b++){
          u32 wv=weight[j][b/2];
          a[m][b]=fmaf(xv,__uint_as_float(b%2?(wv&0xffff0000):(wv<<16)),a[m][b]);
        }
      }
    }
  }

  __shared__ float partial[S][64*V];
  #pragma unroll
  for(int m=0;m<M;m++){
    #pragma unroll
    for(int b=0;b<V;b++) partial[wave][lane*V+b]=a[m][b];
    __syncthreads();
    if(wave==0 && valid){
      #pragma unroll
      for(int b=0;b<V;b++){
        float total=partial[0][lane*V+b];
        #pragma unroll
        for(int i=1;i<S;i++) total+=partial[i][lane*V+b];
        y[m*N+col+b]=rne(total);
      }
    }
    __syncthreads();
  }
}
template<int M,int V,int S,int P>
void launch(const u16*x,const u16*w,u16*y,float*z,hipStream_t st){
  hipLaunchKernelGGL((vector_kn<M,V,S,P>),dim3((N+V*64-1)/(V*64)),dim3(64*S),0,st,x,w,y);
}
extern "C" int run(const u16*x,const u16*w,u16*y,float*z,int m,int cfg,hipStream_t st){
#define SEL(M,V,S,P,C) if(m==M && cfg==C) {launch<M,V,S,P>(x,w,y,z,st);return hipGetLastError();}
#define CFG(V,S,P,C) SEL(1,V,S,P,C) SEL(2,V,S,P,C) SEL(3,V,S,P,C) SEL(4,V,S,P,C)
  CFG(4,2,4,0) CFG(4,4,4,1) CFG(4,4,8,2) CFG(4,8,8,3)
  CFG(8,2,4,4) CFG(8,4,8,5) CFG(8,8,8,6) CFG(8,8,16,7)
  CFG(4,4,16,8) CFG(8,4,16,9)
  return -1;
}
