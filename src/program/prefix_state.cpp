#include "strata/program/prefix_state.hpp"
#include "strata/kernels/qsa.hpp"
#include <cstring>
#include <stdexcept>
namespace strata::program {
namespace {
struct View {void* ptr; size_t bytes; int64_t row_bytes;};
std::vector<View> views(const core::SessionState& ss,const core::ModelGeometry& g,const core::MtpDrafter& mtp,int64_t root) {
    if(root<=0 || root>ss.max_cells)throw std::runtime_error("invalid prefix length");
    auto s=kernels::qsa_real_shapes();
    const int64_t rows=((root+s.page_size-1)/s.page_size)*s.page_size*g.n_head_kv;
    std::vector<View> v;
    for(int64_t i=0;i<=g.n_qsa_layers();++i) {
        const auto& st=i==g.n_qsa_layers()?mtp.kv_state():ss.qsa_states[i];
        if(!st.kv_int8 || st.kv_q4 || st.kv_mode!=0 || !st.k_q || !st.v_q)
            throw std::runtime_error("persistent prefix v1 requires resident INT8 KV and MTP");
        for(const auto& item:std::vector<std::pair<void*,int64_t>>{{st.k_q,g.head_dim},{st.v_q,g.head_dim},
                {st.k_scale,g.head_dim/64*2},{st.v_scale,g.head_dim/64*2}})
            v.push_back({item.first,(size_t)(rows*item.second),item.second});
        if(i<g.n_qsa_layers())v.push_back({st.idx_pooled,(size_t)(root/s.idx_block*g.idx_key_dim*4),0});
    }
    return v;
}
bool copy_ok(cudaError_t e,std::string& err) {
    if(e==cudaSuccess)return true;
    err=cudaGetErrorString(e);
    return false;
}
}
bool prefix_state_capture(const core::SessionState& ss,const core::ModelGeometry& g,const core::MtpDrafter& mtp,
                          int64_t root,std::vector<PrefixBlob>& blobs,std::string& err,
                          const std::function<bool()>& cancelled,uint64_t max_bytes) {
    try {
        const auto v=views(ss,g,mtp,root);
        uint64_t total=0;
        for (const auto& b:blobs) total+=b.size();
        for (const auto& a:v) total+=a.bytes;
        if (total>max_bytes) throw std::runtime_error("state exceeds staging byte limit");
        if (cancelled && cancelled()) throw std::runtime_error("cancelled");
        if(!copy_ok(cudaDeviceSynchronize(),err))return false;
        auto s=kernels::qsa_real_shapes();
        for(const auto& a:v) {
            if (cancelled && cancelled()) throw std::runtime_error("cancelled");
            PrefixBlob b(a.bytes);
            if(a.bytes && !copy_ok(cudaMemcpy(b.data(),a.ptr,a.bytes,cudaMemcpyDeviceToHost),err))return false;
            // Never persist stale cells beyond the requested system prefix in its partially filled last page.
            if(a.row_bytes && root%s.page_size) {
                const int64_t page=root/s.page_size,tail=root%s.page_size;
                for(int64_t h=0;h<g.n_head_kv;++h) {
                    size_t off=(size_t)(((page*g.n_head_kv+h)*s.page_size+tail)*a.row_bytes);
                    std::memset(b.data()+off,0,(size_t)((s.page_size-tail)*a.row_bytes));
                }
            }
            blobs.push_back(std::move(b));
        }
        return true;
    }catch(const std::exception& e){err=e.what();return false;}
}
bool prefix_state_validate(const core::SessionState& ss,const core::ModelGeometry& g,const core::MtpDrafter& mtp,
                           int64_t root,const std::vector<PrefixBlob>& blobs,size_t offset,std::string& err) {
    try {
        auto v=views(ss,g,mtp,root);
        if(offset>blobs.size() || blobs.size()-offset!=v.size())throw std::runtime_error("wrong prefix state record count");
        for(size_t i=0;i<v.size();++i)
            if(blobs[offset+i].size()!=v[i].bytes)throw std::runtime_error("wrong prefix state record size");
        return true;
    }catch(const std::exception& e){err=e.what();return false;}
}
bool prefix_state_restore(const core::SessionState& ss,const core::ModelGeometry& g,const core::MtpDrafter& mtp,
                          int64_t root,const std::vector<PrefixBlob>& blobs,size_t offset,std::string& err) {
    if(!prefix_state_validate(ss,g,mtp,root,blobs,offset,err))return false;
    const auto v=views(ss,g,mtp,root);
    for(size_t i=0;i<v.size();++i)
        if(v[i].bytes && !copy_ok(cudaMemcpy(v[i].ptr,blobs[offset+i].data(),v[i].bytes,cudaMemcpyHostToDevice),err))return false;
    return copy_ok(cudaDeviceSynchronize(),err);
}
}
