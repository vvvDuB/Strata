// No GPU claims: independent scalar arithmetic, packing and RHT invariants.
#include "strata/kernels/kv_nvfp4_codec.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
namespace n=strata::kernels::nvfp4;
static int checks;
static void need(bool b,const char* s){++checks;if(!b){std::fprintf(stderr,"FAIL: %s\n",s);std::exit(1);}}
static float fp4_ref(int c){const float v[]={0,.5f,1,1.5f,2,3,4,6};return c&8?-v[c&7]:v[c&7];}
static float fp8_ref(int c){int e=c>>3,m=c&7;return e?std::ldexp(1.0f+m/8.0f,e-7):m/512.0f;}
static int nearest(float x,const std::vector<float>& levels){
    int out=0;double best=1e300;for(size_t i=0;i<levels.size();++i){double d=std::fabs(double(x)-levels[i]);
        if(d<best||(d==best&&!(i&1))){best=d;out=int(i);}}return out;
}
int main(){
    need(sizeof(n::Row)==148&&alignof(n::Row)==4,"wire layout and alignment");
    std::vector<float> l4,l8;for(int i=0;i<8;++i)l4.push_back(fp4_ref(i));for(int i=0;i<127;++i)l8.push_back(fp8_ref(i));
    for(int i=0;i<16;++i){need(n::bits(n::e2m1_decode(i))==n::bits(fp4_ref(i)),"E2M1 every code including signed zero");
        need(n::e2m1_encode(fp4_ref(i))==i,"E2M1 roundtrip every code");need(n::twice_e2m1(i)==2*fp4_ref(i),"exact twice-FP4 integer");}
    for(int i=0;i<127;++i){need(n::e4m3_decode(i)==l8[i],"E4M3 exhaustive decode");need(n::e4m3_encode_scale(l8[i])==i,"E4M3 exhaustive roundtrip");}
    for(int i=0;i<126;++i){float mid=(l8[i]+l8[i+1])/2;for(float x:{std::nextafter(mid,0.f),mid,std::nextafter(mid,1000.f)})
        need(n::e4m3_encode_scale(x)==nearest(x,l8),"E4M3 midpoint neighbours independent nearest-even oracle");}
    for(int i=0;i<7;++i){float mid=(l4[i]+l4[i+1])/2;for(float x:{std::nextafter(mid,0.f),mid,std::nextafter(mid,1000.f)})
        for(float sign:{1.f,-1.f})need(n::e2m1_encode(sign*x)==(nearest(x,l4)|(sign<0?8:0)),"E2M1 signed midpoint neighbours");}
    std::mt19937 rng(214);std::normal_distribution<float> normal(0,1);std::uniform_real_distribution<float> u8(0,500),u4(0,8);
    for(int i=0;i<30000;++i){float a=u8(rng),b=u4(rng);need(n::e4m3_encode_scale(a)==nearest(a,l8),"random E4M3 oracle");
        need(n::e2m1_encode(b)==nearest(b,l4),"random E2M1 oracle");}
    for(int basis=0;basis<256;++basis){std::array<float,256>x{};x[basis]=1;n::rotate(x.data());double norm=0;
        for(float a:x){need(std::fabs(a)==1.f/16,"RHT basis magnitude");norm+=a*a;}need(norm==1,"RHT orthonormal basis norm");
        n::rotate(x.data(),true);for(int j=0;j<256;++j)need(x[j]==(j==basis?1.f:0.f),"RHT true inverse (D H, not H D)");}
    double sumse=0,sume=0,maxnmse=0;
    for(int trial=0;trial<512;++trial){std::array<float,256>x,y,r,z,q;
        const float scale=std::ldexp(1.f,trial%41-20);
        for(int j=0;j<256;++j){x[j]=normal(rng)*scale;y[j]=normal(rng);}
        if(trial%7==0)x[trial%256]*=80;
        r=x;q=y;n::rotate(r.data());n::rotate(q.data());double dot=0,rdot=0,norm=0,rnorm=0;
        for(int j=0;j<256;++j){dot+=double(x[j])*y[j];rdot+=double(r[j])*q[j];norm+=double(x[j])*x[j];rnorm+=double(r[j])*r[j];}
        need(std::fabs(norm-rnorm)<1e-6*norm,"RHT preserves squared norm");
        need(std::fabs(dot-rdot)<2e-5*std::sqrt(norm*256),"RHT preserves QK dot product");
        auto row=n::encode(x.data()), row2=n::encode_rotated(r.data());need(!std::memcmp(&row,&row2,sizeof row),"deterministic rotate/pack path");
        need(row.tensor_scale>0&&n::finite(row.tensor_scale),"normal dynamic tensor scale");
        for(int b=0;b<16;++b){need(row.scales[b]<127,"finite nonnegative FP8 scale");
            float bm=0;for(int j=0;j<16;++j)bm=std::max(bm,std::fabs(r[b*16+j]));
            need(row.scales[b]==nearest((bm/row.tensor_scale)/6,l8),"rounded block scales independent oracle");
            const float sc=fp8_ref(row.scales[b]);for(int j=0;j<16;++j){int d=b*16+j;float v=sc?(r[d]/row.tensor_scale)/sc:0;
                int exp=nearest(std::fabs(v),l4)|(std::signbit(v)?8:0);int actual=(row.codes[d/2]>>((d&1)*4))&15;
                need(actual==exp,"nibbles use rounded scales and correct even/odd order");}}
        n::decode_row(row,z.data());double se=0;for(int j=0;j<256;++j){need(n::finite(z[j]),"finite reconstructed activation");se+=std::pow(double(x[j])-z[j],2);}
        sumse+=se;sume+=norm;maxnmse=std::max(maxnmse,se/norm);
        need(se/norm<.04,"synthetic vector NMSE below loose regression bound (not model quality)");
    }
    std::array<float,256>x{},z{};auto zero=n::encode(x.data());
    const n::Row empty{};need(!std::memcmp(&zero,&empty,sizeof zero),"zero row canonical");
    for(float special:{INFINITY,-INFINITY,NAN}){x[7]=special;auto bad=n::encode(x.data());need(std::isnan(bad.tensor_scale),"nonfinite input flagged");
        n::decode_row(bad,z.data());for(float v:z)need(std::isnan(v),"invalid activation not silently erased");}x[7]=0;
    for(float tiny:{std::numeric_limits<float>::denorm_min(),FLT_MIN,1e-30f}){x[37]=tiny;auto row=n::encode(x.data());
        need(row.tensor_scale==0||row.tensor_scale>=FLT_MIN,"scale is not subnormal");n::decode_row(row,z.data());for(float v:z)need(n::finite(v),"tiny values safe");}x[37]=0;
    // Pointer-layout validation without dereferencing fake buffers.
    auto s=strata::kernels::qsa_real_shapes();s.page_size=4;
    strata::kernels::QsaAttnPools p;uint8_t k[148]{},v[148]{};int32_t table[1]{};
    p.k_nvfp4=k;p.v_nvfp4=v;p.page_table=table;p.nvfp4_pages=1;p.nvfp4_max_cells=4;
    need(strata::kernels::qsa_nvfp4_pools_valid(p,s),"valid distinct NVFP4 pools");
    p.k_q=reinterpret_cast<int8_t*>(k);need(!strata::kernels::qsa_nvfp4_pools_valid(p,s),"mixed format rejected");p.k_q=nullptr;
    p.v_nvfp4=k;need(!strata::kernels::qsa_nvfp4_pools_valid(p,s),"aliased K/V rejected");p.v_nvfp4=v;
    p.nvfp4_max_cells=5;need(!strata::kernels::qsa_nvfp4_pools_valid(p,s),"insufficient pages rejected");
    std::printf("PASS: %d host codec checks. Synthetic weighted NMSE=%.7f, max=%.7f. NOT a model/GPU benchmark.\n",checks,sumse/sume,maxnmse);
}
