// Real-device parity: production append/rotate/gather/decode/prompt, independent
// CPU attention over the EXACT stored quantized values (not a quality benchmark).
#include "strata/kernels/kv_nvfp4.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_prompt_attn.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>
namespace k=strata::kernels;namespace n=k::nvfp4;
static int checks=0,prompt_calls=0;
static void ck(cudaError_t e){if(e!=cudaSuccess){std::fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e));std::exit(2);}}
static void need(bool x,const char* name){++checks;if(!x){std::fprintf(stderr,"FAIL: %s\n",name);std::exit(1);}}
template<class T>struct Device{
    unsigned char* raw=nullptr;T* p=nullptr;size_t size;
    explicit Device(size_t count):size(count){ck(cudaMalloc(&raw,size*sizeof(T)+128));p=reinterpret_cast<T*>(raw+64);ck(cudaMemset(raw,0xa5,size*sizeof(T)+128));}
    explicit Device(const std::vector<T>& v):Device(v.size()){up(v);}
    ~Device(){if(raw)cudaFree(raw);}Device(const Device&)=delete;Device&operator=(const Device&)=delete;
    void up(const std::vector<T>& v){need(v.size()==size,"upload geometry");ck(cudaMemcpy(p,v.data(),size*sizeof(T),cudaMemcpyHostToDevice));}
    std::vector<T> down(){std::vector<T> v(size);ck(cudaMemcpy(v.data(),p,size*sizeof(T),cudaMemcpyDeviceToHost));return v;}
    void guard(){unsigned char a[64],b[64];ck(cudaMemcpy(a,raw,64,cudaMemcpyDeviceToHost));ck(cudaMemcpy(b,raw+64+size*sizeof(T),64,cudaMemcpyDeviceToHost));
        for(int i=0;i<64;++i)need(a[i]==0xa5&&b[i]==0xa5,"device allocation canaries");}
};
static uint16_t half_bits(float x){__half h=__float2half_rn(x);uint16_t b;std::memcpy(&b,&h,2);return b;}
static void near(const std::vector<float>& got,const std::vector<float>& ref,double tol,const char* label){
    need(got.size()==ref.size(),"comparison geometry");double err=0,mag=0;
    for(size_t i=0;i<got.size();++i){need(std::isfinite(got[i]),"finite GPU output");err=std::max(err,std::fabs(double(got[i])-ref[i]));mag=std::max(mag,std::fabs(double(ref[i])));}
    if(err>tol*(1+mag)){std::fprintf(stderr,"%s max_abs=%g scale=%g limit=%g\n",label,err,mag,tol*(1+mag));std::exit(1);}++checks;
}
static void rotations(){
    for(int rows:{1,3,4,5,17}){std::vector<float> x(rows*256),ref;for(size_t i=0;i<x.size();++i)x[i]=std::sin(float(i)*.13f)*7;
        ref=x;for(int r=0;r<rows;++r)n::rotate(ref.data()+r*256);
        Device<float> input(x),output(x.size());k::nvfp4_rotate_cuda(input.p,output.p,rows,false,nullptr);
        near(output.down(),ref,1e-6,"RHT forward");need(input.down()==x,"out-of-place RHT leaves input untouched");
        k::nvfp4_rotate_cuda(output.p,output.p,rows,true,nullptr);near(output.down(),x,3e-6,"RHT inverse in place");input.guard();output.guard();}
}
static std::vector<float> attention(const std::vector<n::Row>& keys,const std::vector<n::Row>& vals,
    const std::vector<int32_t>& table,const std::vector<int32_t>& ids,const std::vector<int32_t>& steps,
    const std::vector<float>& q,int PS,int cells,int cap,int nq){
    constexpr int NH=24,NKV=2,HD=256;std::vector<float> out(nq*NH*HD,0);
    for(int t=0;t<nq;++t)for(int h=0;h<NH;++h){int width=std::max(0,std::min(cap,steps[t*k::kStepCount+k::kStepWidth]));
        std::vector<double> scores(width,-INFINITY);std::vector<int> rows(width,-1);double m=-INFINITY,sum=0;
        for(int j=0;j<width;++j){int cell=ids[t*cap+j];if(cell<0||cell>=cells)continue;int page=table[cell/PS];if(page<0||page>=int(table.size()))continue;
            int r=(page*NKV+h/12)*PS+cell%PS;rows[j]=r;double z=0;
            for(int d=0;d<HD;++d)z+=double(q[(t*NH+h)*HD+d])*n::decode(keys[r],d);
            scores[j]=z/16;m=std::max(m,scores[j]);}
        for(int j=0;j<width;++j)if(rows[j]>=0){scores[j]=std::exp(scores[j]-m);sum+=scores[j];}
        if(sum>0)for(int d=0;d<HD;++d){double z=0;for(int j=0;j<width;++j)if(rows[j]>=0)z+=(scores[j]/sum)*n::decode(vals[rows[j]],d);
            out[(t*NH+h)*HD+d]=float(z);}
    }return out;
}
static void pools_and_attention(int PS,int cells,int cap){
    constexpr int NH=24,NKV=2,HD=256,NQ=5;int pages=(cells+PS-1)/PS,rows=pages*PS*NKV;
    k::QsaShapes s=k::qsa_real_shapes();s.page_size=PS;
    std::mt19937 rng(350+PS+cells);std::normal_distribution<float> normal;
    std::vector<int32_t> table(pages);std::iota(table.begin(),table.end(),0);std::shuffle(table.begin(),table.end(),rng);
    std::vector<float> K(cells*NKV*HD),V(K.size());for(float&x:K)x=normal(rng);for(float&x:V)x=normal(rng)*.7f;
    // Mixed zero, outlier and very different row/block scales.
    for(int d=0;d<HD;++d){K[d]=0;V[d]=0;}if(cells>1){K[2*HD+7]=35;V[3*HD+91]=-49;}
    Device<float> dK(K),dV(V);Device<int32_t> tab(table);Device<n::Row> pk(rows),pv(rows);
    k::kv_append_nvfp4((uint8_t*)pk.p,(uint8_t*)pv.p,tab.p,0,cells,dK.p,dV.p,cells,s,nullptr);ck(cudaDeviceSynchronize());
    auto keys=pk.down(),vals=pv.down();
    for(int t=0;t<cells;++t)for(int h=0;h<NKV;++h){int r=(table[t/PS]*NKV+h)*PS+t%PS;
        for(int side=0;side<2;++side){const auto&actual=side?vals[r]:keys[r];const float*src=(side?V:K).data()+(t*NKV+h)*HD;
            auto expected=n::encode(src);double e=0,mag=0;
            for(int d=0;d<HD;++d){float a=n::decode(actual,d),b=n::decode(expected,d);e=std::max(e,std::fabs(double(a)-b));mag=std::max(mag,std::fabs(double(b)));}
            // CPU/GPU floating-point divisions may change a midpoint decision;
            // demand exact scales and nearly exact reconstructed rows in this fixture.
            need(e<1e-5*(1+mag),"GPU fused RHT encode vs scalar reference");}}
    // Every unused cell in a partially filled page must retain its canary bytes.
    if(cells%PS)for(int h=0;h<NKV;++h)for(int tail=cells%PS;tail<PS;++tail){int r=(table.back()*NKV+h)*PS+tail;
        for(const auto* a:{&keys[r],&vals[r]}){const auto*b=reinterpret_cast<const uint8_t*>(a);for(size_t j=0;j<sizeof(n::Row);++j)need(b[j]==0xa5,"partial append untouched tail");}}
    need(dK.down()==K&&dV.down()==V,"append does not mutate K/V activations");
    // Capture one step append, then change only device step and source buffers.
    std::vector<int32_t> step(k::kStepCount,0);Device<int32_t> st(step);Device<float> oneK(NKV*HD),oneV(NKV*HD);
    cudaStream_t stream;cudaGraph_t graph;cudaGraphExec_t exec;ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
    k::kv_append_nvfp4_step((uint8_t*)pk.p,(uint8_t*)pv.p,tab.p,st.p,oneK.p,oneV.p,cells,s,stream);
    ck(cudaStreamEndCapture(stream,&graph));ck(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    for(int pos:{0,cells/2,cells-1}){step[k::kStepPos]=pos;st.up(step);
        std::vector<float>a(K.begin()+pos*NKV*HD,K.begin()+(pos+1)*NKV*HD),b(V.begin()+pos*NKV*HD,V.begin()+(pos+1)*NKV*HD);oneK.up(a);oneV.up(b);
        ck(cudaGraphLaunch(exec,stream));ck(cudaStreamSynchronize(stream));}
    auto stepKeys=pk.down(),stepVals=pv.down();need(!std::memcmp(keys.data(),stepKeys.data(),rows*sizeof(n::Row)),"captured append reads changing device position");
    need(!std::memcmp(vals.data(),stepVals.data(),rows*sizeof(n::Row)),"step/batch V bitwise parity");
    ck(cudaGraphExecDestroy(exec));ck(cudaGraphDestroy(graph));ck(cudaStreamDestroy(stream));
    // One deliberately missing and one out-of-range page: all consumers must mask.
    if(pages>3){table[1]=-1;table[2]=pages;}tab.up(table);
    std::vector<int32_t> ids(NQ*cap),steps(NQ*k::kStepCount,0);
    for(int t=0;t<NQ;++t){for(int j=0;j<cap;++j)ids[t*cap+j]=(j*37+t*3)%cells;
        steps[t*k::kStepCount+k::kStepWidth]=(t==0?0:t==1?-2:t==2?std::max(1,cap-3):t==3?cap+3:cap);}
    if(cap>2){ids[2*cap]=-1;ids[3*cap+1]=cells;ids[4*cap+2]=INT32_MAX;}
    std::vector<float> q(NQ*NH*HD);for(float&x:q)x=normal(rng)*1.1f;for(int r=0;r<NQ*NH;++r)n::rotate(q.data()+r*HD);
    Device<int32_t> di(ids),ds(steps);Device<float> dq(q),out(q.size()),prompt(q.size());
    Device<float> scratch(NQ*k::qsa_decode_attn_scratch_floats(cap,s));
    k::QsaAttnPools p;p.page_table=tab.p;p.k_nvfp4=(uint8_t*)pk.p;p.v_nvfp4=(uint8_t*)pv.p;p.nvfp4_max_cells=cells;p.nvfp4_pages=pages;
    const auto ref=attention(keys,vals,table,ids,steps,q,PS,cells,cap,NQ);
    k::qsa_decode_attn_batch(dq.p,p,di.p,ds.p,cap,s,scratch.p,out.p,NQ,nullptr);near(out.down(),ref,3e-5,"batched decode FP64 oracle");
    for(int t=0;t<NQ;++t)k::qsa_decode_attn_step(dq.p+t*NH*HD,p,di.p+t*cap,ds.p+t*k::kStepCount,cap,s,scratch.p,out.p+t*NH*HD,nullptr);
    near(out.down(),ref,3e-5,"step decode FP64 oracle");
    if(k::qsa_prompt_attn_batch(dq.p,p,di.p,ds.p,cap,s,prompt.p,NQ,nullptr)){++prompt_calls;near(prompt.down(),ref,5e-5,"tensorcore prompt FP64 oracle");}
    else {std::fprintf(stderr,"FAIL: NVFP4 prompt path not dispatched; this test requires SM80+\n");std::exit(1);}
    // Output inverse is outside attention, before the gate. Verify both readers.
    auto invref=ref;for(int r=0;r<NQ*NH;++r)n::rotate(invref.data()+r*HD,true);
    k::nvfp4_rotate_cuda(out.p,out.p,NQ*NH,true,nullptr);near(out.down(),invref,5e-5,"decode output inverse");
    k::nvfp4_rotate_cuda(prompt.p,prompt.p,NQ*NH,true,nullptr);near(prompt.down(),invref,7e-5,"prompt output inverse");
    Device<uint16_t> gatheredK(cap*NKV*HD),gatheredV(cap*NKV*HD);
    k::kv_gather_nvfp4_step(p.k_nvfp4,p.v_nvfp4,tab.p,di.p+4*cap,ds.p+4*k::kStepCount,cap,cells,s,gatheredK.p,gatheredV.p,nullptr);
    auto gK=gatheredK.down(),gV=gatheredV.down();
    for(int j=0;j<cap;++j){int cell=ids[4*cap+j],page=cell>=0&&cell<cells?table[cell/PS]:-1;bool valid=page>=0&&page<pages;
        for(int h=0;h<NKV;++h)for(int d=0;d<HD;++d){int r=valid?(page*NKV+h)*PS+cell%PS:0,idx=(j*NKV+h)*HD+d;
            need(gK[idx]==half_bits(valid?n::decode(keys[r],d):0),"gather K only selected valid rows");need(gV[idx]==half_bits(valid?n::decode(vals[r],d):0),"gather V only selected valid rows");}}
    // Decode graph replay with different widths catches accidentally captured host counts.
    ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
    k::qsa_decode_attn_step(dq.p,p,di.p,ds.p,cap,s,scratch.p,out.p,stream);
    ck(cudaStreamEndCapture(stream,&graph));ck(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    for(int width:{0,1,cap}){steps[k::kStepWidth]=width;ds.up(steps);ck(cudaGraphLaunch(exec,stream));ck(cudaStreamSynchronize(stream));
        auto all=out.down();all.resize(NH*HD);auto r=attention(keys,vals,table,ids,steps,q,PS,cells,cap,1);near(all,r,3e-5,"graph decode dynamic width");}
    ck(cudaGraphExecDestroy(exec));ck(cudaGraphDestroy(graph));ck(cudaStreamDestroy(stream));
    pk.guard();pv.guard();dK.guard();dV.guard();tab.guard();st.guard();oneK.guard();oneV.guard();dq.guard();
    out.guard();prompt.guard();scratch.guard();di.guard();ds.guard();gatheredK.guard();gatheredV.guard();
    std::printf("PASS fixture: page=%d cells=%d selected-cap=%d\n",PS,cells,cap);
}
int main(){int count=0;auto e=cudaGetDeviceCount(&count);if(e!=cudaSuccess||count<1){std::puts("SKIP: no CUDA GPU");return 77;}
    cudaDeviceProp p{};ck(cudaGetDeviceProperties(&p,0));if(p.major<8){std::puts("SKIP: prompt parity requires SM80+");return 77;}
    std::printf("GPU: %s sm%d%d\n",p.name,p.major,p.minor);
    rotations();for(int ps:{1,4,64})for(int cap:{1,31,32,33,63,64,65})pools_and_attention(ps,131,cap);
    pools_and_attention(4,4099,2051);ck(cudaDeviceSynchronize());need(prompt_calls==22,"every prompt fixture exercised");
    std::printf("PASS: %d real-GPU checks, %d prompt fixtures. This does not measure model quality or speed.\n",checks,prompt_calls);
}
