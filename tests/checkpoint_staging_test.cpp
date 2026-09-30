#include "strata/core/checkpoint_staging.hpp"
#include <array>
#include <cstdio>
#include <limits>
using namespace strata::core;
struct Mock {
    std::vector<uint8_t> stage;
    std::vector<CheckpointCopy> queued;
    bool allocation=true, event=true, waiting=true;
    int fail_copy=0, copies=0, drains=0, releases=0, waits=0;
    uint8_t* data() { return stage.data(); }
    bool reserve(size_t n,std::string&) { if(!allocation)return false; stage.resize(n); return true; }
    bool enqueue(void* d,const void* s,size_t n,std::string&) {
        if(++copies==fail_copy)return false; queued.push_back({s,d,n}); return true;
    }
    bool record(std::string&) { return event; }
    void drain() { ++drains; for(auto c:queued)std::memcpy(c.destination,c.source,c.bytes);queued.clear(); }
    bool wait(std::string&) { ++waits;if(!waiting)return false;drain();return true; }
    void release() { ++releases;stage.clear(); }
};
#define CHECK(x) do { if(!(x)){std::fprintf(stderr,"line %d: %s\n",__LINE__,#x);return 1;} } while(0)
int main() {
    std::array<std::vector<uint8_t>,5> src, dst;
    std::vector<CheckpointCopy> spans;
    for(size_t i=0;i<5;++i){src[i].assign(i+1,uint8_t(i+10));dst[i].assign(i+1,0);spans.push_back({src[i].data(),dst[i].data(),i+1});}
    spans.push_back({nullptr,nullptr,0});
    std::string e;
    Mock m;
    {
        CheckpointStaging<Mock> c(m);
        CHECK(c.finish(e)); CHECK(c.enqueue(spans,e)==CheckpointEnqueue::queued);
        CHECK(c.pending()); CHECK(dst[0][0]==0);
        CHECK(c.enqueue(spans,e)==CheckpointEnqueue::failed);
        CHECK(c.finish(e)); CHECK(!c.pending()); CHECK(src==dst);
        CHECK(c.finish(e)); CHECK(m.waits==1);
        // Destruction while cancelled drains DMA but never publishes buffers.
        for(auto& v:dst)std::fill(v.begin(),v.end(),0);
        CHECK(c.enqueue(spans,e)==CheckpointEnqueue::queued);
    }
    CHECK(dst[0][0]==0); CHECK(m.drains==2 && m.releases==1);
    for(int mode=0;mode<4;++mode){
        Mock t; CheckpointStaging<Mock> c(t);
        if(mode==0)t.allocation=false;
        if(mode==1)t.fail_copy=3;
        if(mode==2)t.event=false;
        if(mode==3)t.waiting=false;
        const auto result=c.enqueue(spans,e);
        if(mode==0){CHECK(result==CheckpointEnqueue::unavailable);CHECK(t.copies==0);}
        else if(mode==3){CHECK(result==CheckpointEnqueue::queued);CHECK(!c.finish(e));CHECK(t.drains==1);}
        else {CHECK(result==CheckpointEnqueue::failed);CHECK(t.drains==1);}
        CHECK(!c.pending()); CHECK(dst[0][0]==0);
    }
    { Mock t;CheckpointStaging<Mock> c(t);
      CHECK(c.enqueue({{nullptr,nullptr,1}},e)==CheckpointEnqueue::failed);
      CHECK(c.enqueue({{src[0].data(),dst[0].data(),std::numeric_limits<size_t>::max()},spans[0]},e)==CheckpointEnqueue::failed);
      CHECK(c.enqueue({},e)==CheckpointEnqueue::unavailable);CHECK(t.copies==0);
    }
    std::puts("checkpoint_staging: lifecycle, five buffers, cancellation, overflow and fault checks passed");
}
