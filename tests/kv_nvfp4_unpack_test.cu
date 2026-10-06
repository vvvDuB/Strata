// Exhaustive real-GPU and host parity for four NVFP4 -> twice-E2M1 INT8 codes.
#include "strata/kernels/kv_nvfp4_codec.hpp"
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>
#if !__has_include("strata/kernels/kv_nvfp4_unpack4.cuh")
int main() { std::fprintf(stderr,"FAIL: vector NVFP4 unpack helper is missing\n"); return 1; }
#else
#include "strata/kernels/kv_nvfp4_unpack4.cuh"
#include <cuda_runtime.h>
namespace n=strata::kernels::nvfp4;
__global__ void run(const uint32_t* x,uint32_t* y,int N){
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<N)y[i]=n::unpack_twice_e2m1_4(x[i]);
}
int main(){
 std::vector<uint32_t> inputs;
 for(uint32_t i=0;i<65536;++i)inputs.push_back(i);
 std::mt19937 rng(442);for(int i=0;i<100000;++i)inputs.push_back(rng());
 std::vector<uint32_t> expected(inputs.size()),actual(inputs.size());
 for(size_t i=0;i<inputs.size();++i){
  for(int j=0;j<4;++j)expected[i]|=uint32_t(uint8_t(n::twice_e2m1(inputs[i]>>(4*j))))<<(8*j);
  if(n::unpack_twice_e2m1_4(inputs[i])!=expected[i]){std::fprintf(stderr,"FAIL host unpack %zu\n",i);return 1;}
 }
 int devices=0;auto err=cudaGetDeviceCount(&devices);if(err!=cudaSuccess||devices<1){std::puts("SKIP GPU unavailable");return 77;}
 uint32_t *x=nullptr,*y=nullptr;size_t bytes=inputs.size()*4;
 auto ck=[](cudaError_t e){if(e!=cudaSuccess){std::fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e));std::exit(2);}};
 ck(cudaMalloc(&x,bytes));ck(cudaMalloc(&y,bytes));ck(cudaMemcpy(x,inputs.data(),bytes,cudaMemcpyHostToDevice));
 run<<<(inputs.size()+127)/128,128>>>(x,y,inputs.size());ck(cudaGetLastError());ck(cudaMemcpy(actual.data(),y,bytes,cudaMemcpyDeviceToHost));
 for(size_t i=0;i<inputs.size();++i)if(actual[i]!=expected[i]){std::fprintf(stderr,"FAIL GPU unpack %zu %08x!=%08x\n",i,actual[i],expected[i]);return 1;}
 ck(cudaFree(x));ck(cudaFree(y));std::printf("PASS host + GPU: %zu packed words, %zu signed code checks (negative zero included)\n",inputs.size(),inputs.size()*4);
}
#endif
