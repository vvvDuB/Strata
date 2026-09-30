#include "strata/platform/conversation_store.hpp"
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <sstream>
#include <stdexcept>
using namespace strata::core;
using namespace strata::platform;
#define CHECK(x) do { if(!(x)){std::fprintf(stderr,"line %d: %s (%s)\n",__LINE__,#x,error.c_str());return 1;} } while(0)
int main() {
    ConversationIdentity id{}; std::string error;
    SavedConversation input; input.live.ids={1,2,3}; input.live.gdn.resize(3*1024*1024,42);
    input.checkpoints.push_back(input.live); input.checkpoints.back().ids.resize(2);
    std::ostringstream out(std::ios::binary);
    CHECK(conversation_file_write(out,input,id,error));
    for(int when : {1,2,4,100000}) {
        int calls=0;
        auto cancel=[&] { if(++calls==when)throw std::runtime_error("cancelled"); };
        SavedConversation target;target.live.ids={999};
        std::istringstream in(out.str(),std::ios::binary);
        bool ok=conversation_file_read(in,id,32<<20,64<<20,0,target,error,cancel);
        if(when<100000){CHECK(!ok); CHECK(error=="cancelled");CHECK(target.live.ids==std::vector<int32_t>{999});}
        else {CHECK(ok); CHECK(target.live.ids==input.live.ids);}
    }
    // Also cancel in finish(): publication remains atomic even for a tiny image.
    { SavedConversation tiny; tiny.live.ids={1};std::ostringstream encoded(std::ios::binary);
      CHECK(conversation_file_write(encoded,tiny,id,error));
      SavedConversation target;target.live.ids={999};int calls=0;
      std::istringstream in(encoded.str(),std::ios::binary);
      CHECK(!conversation_file_read(in,id,1<<20,2<<20,0,target,error,[&]{if(++calls==1)throw std::runtime_error("cancelled");}));
      CHECK(target.live.ids==std::vector<int32_t>{999});
    }
    const auto root=std::filesystem::temp_directory_path()/
        ("strata-cancel-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    struct Clean { std::filesystem::path p;~Clean(){std::error_code e;std::filesystem::remove_all(p,e);} } clean{root};
    ConversationStore store;CHECK(store.open(root,id,32<<20,4,error));
    CHECK(store.put(input,error));
    int calls=0;store.set_progress([&]{if(++calls==3)throw std::runtime_error("cancelled");});
    CHECK(!store.put(input,error));CHECK(error=="cancelled");
    int snaps=0;
    for(const auto& e:std::filesystem::directory_iterator(root/"strata-conversations-v1")){
      CHECK(e.path().extension()!=".tmp"); if(e.path().extension()==".snap")++snaps;
    }
    CHECK(snaps==1);
    store.set_progress([]{throw std::runtime_error("cancelled");});
    ConversationStore::Candidate hit;
    CHECK(!store.best({1,2,3,4},{},true,32<<20,{},hit,error));CHECK(error=="cancelled");
    store.set_progress({});CHECK(store.best({1,2,3,4},{},true,32<<20,{},hit,error));CHECK(hit.match.tokens==3);
    std::puts("conversation I/O cancellation: atomic read, finalization, interrupted spill, cleanup and recovery passed");
}
