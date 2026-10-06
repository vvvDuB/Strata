// Exercise the exact conversion used by the production writer, on a real GPU.
#include "strata/kernels/kv_nvfp4_convert.cuh"
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
namespace n=strata::kernels::nvfp4;
static void ck(cudaError_t e){if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(2);}}
__global__ void convert(const float* a,unsigned char* out,int count,int* mode){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count)out[i]=n::pack_pair_device(a[2*i],a[2*i+1]);
    if(i==0){
#if STRATA_NV4_NATIVE_CONVERT && __CUDA_ARCH__ >= 1000
        *mode=1;
#else
        *mode=0;
#endif
    }
}
int main(){
    int count=0;auto status=cudaGetDeviceCount(&count);
    if(status!=cudaSuccess||count==0){std::puts("SKIP: no CUDA device");return 77;}
    std::vector<float> v{-0.f,0.f,-6.f,6.f,-100.f,100.f,
        std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN(),-std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::denorm_min(),-std::numeric_limits<float>::denorm_min()};
    for(float a:{.25f,.75f,1.25f,1.75f,2.5f,3.5f,5.f})for(float s:{-1.f,1.f})
        for(float x:{std::nextafter(a,0.f),a,std::nextafter(a,10.f)})v.push_back(s*x);
    std::vector<float> pairs;for(float a:v)for(float b:v){pairs.push_back(a);pairs.push_back(b);}
    // Dense additional signed values; exercise pack lanes independently.
    for(int i=-10000;i<=10000;++i){pairs.push_back(i/1024.f);pairs.push_back(-i/2048.f);}
    float* x=nullptr;unsigned char* y=nullptr;int* mode=nullptr;
    ck(cudaMalloc(&x,pairs.size()*4));ck(cudaMalloc(&y,pairs.size()/2));ck(cudaMalloc(&mode,4));
    ck(cudaMemcpy(x,pairs.data(),pairs.size()*4,cudaMemcpyHostToDevice));
    int N=int(pairs.size()/2);convert<<<(N+127)/128,128>>>(x,y,N,mode);ck(cudaGetLastError());ck(cudaDeviceSynchronize());
    std::vector<unsigned char> out(N);int native=0;ck(cudaMemcpy(out.data(),y,N,cudaMemcpyDeviceToHost));
    ck(cudaMemcpy(&native,mode,4,cudaMemcpyDeviceToHost));
    for(int i=0;i<N;++i){int expected=n::e2m1_encode(pairs[2*i])|(n::e2m1_encode(pairs[2*i+1])<<4);
        if(out[i]!=expected){std::fprintf(stderr,"PAIR %d (%g,%g): expected %02x got %02x\n",i,pairs[2*i],pairs[2*i+1],expected,out[i]);return 1;}}
    ck(cudaFree(x));ck(cudaFree(y));ck(cudaFree(mode));
    std::printf("PASS: %d FP4 pairs, native=%d. RNE, NaN/Inf saturation, signed zero, subnormals, nibble order.\n",N,native);
    cudaDeviceProp props{};ck(cudaGetDeviceProperties(&props,0));
    if(props.major>=10&&!native){std::fprintf(stderr,"Native path not exercised: build CUDA >=12.8 for this actual GPU architecture.\n");return 77;}
    return 0;
}
