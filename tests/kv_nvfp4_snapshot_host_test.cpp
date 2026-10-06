// Actual snapshot + prefix-state code with host memcpy substituted for CUDA DMA.
// This tests record geometry, validation, copy extents and prefix tail clearing,
// not GPU correctness. No production code includes host_cuda_stub.
#include "strata/core/conversation_snapshot.hpp"
#include "strata/program/prefix_state.hpp"
#include "strata/kernels/kv_nvfp4.hpp"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
namespace c=strata::core;
namespace k=strata::kernels;
namespace p=strata::program;
namespace strata::kernels {
void kv_stream_reset(const KvStreamMap&,void*) {std::abort();}
void kv_ring_restore(const QsaAttnPools&,const KvHostPools&,int,int64_t,int64_t,int64_t,const QsaShapes&,void*) {std::abort();}
}
namespace strata::core {
strata::kernels::QsaAttnPools qsa_attn_pools(const QsaState&) {std::abort();}
MtpDrafter::~MtpDrafter() = default; // fixture owns no device allocations
}
namespace {
int checks=0;
void require(bool ok,const char* label) {++checks;if(!ok){std::fprintf(stderr,"FAIL: %s\n",label);std::exit(1);}}
struct Fixture {
    c::ModelGeometry g;
    c::QsaState st;
    std::array<std::vector<uint8_t>,5> data;
    explicit Fixture(bool nv4) {
        g.n_layers=4; g.n_head_kv=2;g.head_dim=256;g.idx_key_dim=128;
        st.max_cells=96;st.n_pages=24;st.n_slots=24;st.idx_pooled_rows=26;
        st.kv_int8=!nv4;st.kv_nvfp4=nv4;
        data[0].resize(96*2*(nv4?148:256));data[1].resize(96*2*(nv4?148:256));
        data[2].resize(nv4?0:96*2*8);data[3].resize(nv4?0:96*2*8);data[4].resize(26*128*4);
        if(nv4){st.k_nvfp4=data[0].data();st.v_nvfp4=data[1].data();}
        else {st.k_q=reinterpret_cast<int8_t*>(data[0].data());st.k_scale=reinterpret_cast<uint16_t*>(data[2].data());
              st.v_q=reinterpret_cast<int8_t*>(data[1].data());st.v_scale=reinterpret_cast<uint16_t*>(data[3].data());}
        st.idx_pooled=reinterpret_cast<float*>(data[4].data());
        fill(13);
    }
    void fill(int salt) {for(size_t i=0;i<data.size();++i)for(size_t j=0;j<data[i].size();++j)data[i][j]=uint8_t(salt+i*19+j*7+j/257);}
};
bool same(const c::ConversationKv& a,const c::ConversationKv& b) {
    return a.format==b.format && a.k==b.k && a.v==b.v && a.k_scale==b.k_scale && a.v_scale==b.v_scale && a.pooled==b.pooled;
}
}
int main() {
    for(bool nv4:{false,true}) {
        Fixture f(nv4);std::string error;
        for(int upto:{0,1,3,4,5,63,64,65,96}) for(bool index:{false,true}) {
            f.fill(13); c::ConversationKv a,b,r;
            require(c::conversation_kv_save(a,f.st,f.g,upto,index,error),"capture A");
            require(a.format==(nv4?5:1),"distinct snapshot format");
            require(a.bytes()==c::conversation_kv_bytes(f.st,f.g,upto,index),"capture estimate matches allocation");
            require(a.v_scale.size()==(nv4?0:a.k_scale.size()),"V scales only in INT8");
            const size_t padded=(upto+3)/4*4;
            require(a.v.size()==padded*2*(nv4?148:256),"V extent including partial last page");
            f.fill(177);require(c::conversation_kv_save(b,f.st,f.g,upto,index,error),"capture B");
            require(c::conversation_kv_restore(a,f.st,f.g,upto,index,error),"restore A over B");
            require(c::conversation_kv_save(r,f.st,f.g,upto,index,error),"read restored A");require(same(a,r),"A/B/A exact bytes");
            uint64_t hash=0;require(c::conversation_kv_verify(a,f.st,f.g,upto,index,hash,error),"read-back fingerprint");
            const auto original=f.data;
            auto reject=[&](const c::ConversationKv& bad) {
                size_t before=strata_test_cuda::copies;
                require(!c::conversation_kv_restore(bad,f.st,f.g,upto,index,error),"invalid snapshot rejected");
                require(before==strata_test_cuda::copies && original==f.data,"reject before any destination write");
            };
            auto bad=a;bad.format=nv4?4:5;reject(bad);
            bad=a;bad.v.resize(bad.v.size()+1,0);reject(bad);
            bad=a;bad.head_dim=128;reject(bad);
            if(nv4) {
                for(int mode:{1,2}) {f.st.kv_mode=mode;reject(a);} f.st.kv_mode=0;
                f.st.kv_rot=true;reject(a);f.st.kv_rot=false;
                f.st.kv_hybrid=true;reject(a);f.st.kv_hybrid=false;
                f.st.kv_int8=true;reject(a);f.st.kv_int8=false;
            }
            if(upto) {
                auto wrong=a;wrong.format=nv4?1:5;
                const auto copies=strata_test_cuda::copies;
                require(!c::conversation_kv_save(wrong,f.st,f.g,upto,index,error,upto),"cross-format incremental reuse rejected");
                require(copies==strata_test_cuda::copies,"incremental mismatch rejected before transfer");
                strata_test_cuda::fail_next_copy=true;
                require(!c::conversation_kv_restore(a,f.st,f.g,upto,index,error),"transfer failure is reported");
                require(error.find("test transfer failure")!=std::string::npos,"transfer failure cause preserved");
            }
        }
        // Legacy persistent-prefix records, actual production capture/restore.
        Fixture draft(false);c::MtpDrafter mtp;mtp.kv_state_rw()=draft.st;
        c::SessionState ss;ss.max_cells=96;ss.qsa_states=&f.st;
        for(int root:{1,3,4,5,63,64,65,96}) {
            std::vector<p::PrefixBlob> blobs;
            f.fill(29);draft.fill(31);
            require(p::prefix_state_capture(ss,f.g,mtp,root,blobs,error),"persistent prefix capture");
            require(blobs.size()==(nv4?7:9),"two NVFP4 records or four INT8 records plus indexer and drafter");
            require(blobs[1].size()==size_t((root+3)/4*4)*2*(nv4?148:256),"prefix V record size");

            if(root%4) for(int h=0;h<2;++h) {
                const int rowbytes=nv4?148:256;
                const size_t off=((root/4*2+h)*4+root%4)*rowbytes;
                for(size_t j=off;j<off+size_t(4-root%4)*rowbytes;++j)require(blobs[1][j]==0,"prefix trailing cells cleared");
            }
            f.fill(91);draft.fill(92);
            require(p::prefix_state_restore(ss,f.g,mtp,root,blobs,0,error),"persistent prefix restore");
            std::vector<p::PrefixBlob> restored;
            require(p::prefix_state_capture(ss,f.g,mtp,root,restored,error),"persistent prefix readback");
            require(blobs==restored,"persistent prefix A/B/A exact including cleared tail");
            const auto before=f.data;const auto copies=strata_test_cuda::copies;
            blobs[1].push_back(0);
            require(!p::prefix_state_restore(ss,f.g,mtp,root,blobs,0,error),"prefix wrong V record refused");
            require(before==f.data && copies==strata_test_cuda::copies,"prefix prevalidates before writes");
        }
    }
    std::printf("PASS: %d host-transfer snapshot/prefix checks. NOT a CUDA test.\n",checks);
}
