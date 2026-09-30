// Real CUDA regression: batched PLE must match sequential postops and stay in bounds.
#include "strata/kernels/native_ple_postops.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
namespace k = strata::kernels;
constexpr int N=2560, D=N*4;
void ck(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
struct Buffer {
    float* p; size_t n;
    explicit Buffer(size_t count): n(count) { ck(cudaMalloc(&p,n*4)); }
    ~Buffer(){ cudaFree(p); }
    void set(const std::vector<float>& v){ ck(cudaMemcpy(p,v.data(),n*4,cudaMemcpyHostToDevice)); }
    std::vector<float> get(){ std::vector<float> v(n); ck(cudaMemcpy(v.data(),p,n*4,cudaMemcpyDeviceToHost)); return v; }
};
int main(int argc, char** argv) {
    const int T=argc>1?std::stoi(argv[1]):77;
    if (T <= 0 || T > 4096) throw std::invalid_argument("T must be 1..4096");
    cudaStream_t s; ck(cudaStreamCreate(&s));
    std::vector<float> keys(T*D), hidden(T*D), values(T*N), hist(9*D), gamma(D,1.0f);
    for(size_t i=0;i<keys.size();++i){keys[i]=std::sin(float(i)*0.031f); hidden[i]=std::cos(float(i)*0.023f);}
    for(size_t i=0;i<values.size();++i) values[i]=std::sin(float(i)*0.013f);
    for(size_t i=0;i<hist.size();++i) hist[i]=std::cos(float(i)*0.011f);
    Buffer key(T*D), h(T*D), val(T*N), history(9*D), norm(D), qn(T*D), gated(T*D), gate(T*4);
    Buffer convw(D*2); // 4*D half values in 2*D float storage
    std::vector<__half> cw(4*D,__float2half(0.1f));
    ck(cudaMemcpy(convw.p,cw.data(),cw.size()*2,cudaMemcpyHostToDevice));
    key.set(keys); h.set(hidden); val.set(values); history.set(hist); norm.set(gamma);
    k::PleWeights w{}; w.norm_key=w.norm_query=w.norm_conv=norm.p; w.conv1d_f16=(uint16_t*)convw.p;
    k::native_ple_postops_batch(key.p,h.p,val.p,history.p,w,qn.p,gated.p,gate.p,T,s);
    ck(cudaStreamSynchronize(s));
    const auto actual=h.get(), actual_hist=history.get();
    Buffer sk(D), sh(D), sv(N), sq(D), sg(4), sgd(D), sn(D), sc(D), sr(D), projected(D);
    history.set(hist);
    k::NativePlePostopsBuffers b{sk.p,sq.p,sg.p,sgd.p,sn.p,sc.p,sr.p};
    float maxerr=0;
    for(int t=0;t<T;++t){
        ck(cudaMemcpy(projected.p,keys.data()+t*D,D*4,cudaMemcpyHostToDevice));
        ck(cudaMemcpy(sh.p,hidden.data()+t*D,D*4,cudaMemcpyHostToDevice));
        ck(cudaMemcpy(sv.p,values.data()+t*N,N*4,cudaMemcpyHostToDevice));
        k::native_ple_postops(projected.p,sh.p,sv.p,history.p,w,b,s); ck(cudaStreamSynchronize(s));
        auto expected=sr.get(), normalized=sn.get();
        for(int c=0;c<D;++c){
            if(!std::isfinite(actual[t*D+c])) throw std::runtime_error("nonfinite batch result");
            maxerr=std::max(maxerr,std::abs(expected[c]-actual[t*D+c]));
            for(int r=0;r<8;++r) hist[c*9+r]=hist[c*9+r+1];
            hist[c*9+8]=normalized[c];
        }
        history.set(hist);
    }
    for(size_t i=0;i<hist.size();++i) maxerr=std::max(maxerr,std::abs(hist[i]-actual_hist[i]));
    std::printf("T=%d max absolute error=%g\n",T,maxerr);
    ck(cudaStreamDestroy(s));
    return maxerr>1e-5f?1:0;
}
