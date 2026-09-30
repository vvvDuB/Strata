#include "strata/program/conversation_store.hpp"
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
using namespace strata::program;
namespace fs = std::filesystem;
int main(int argc, char** argv) {
    if (argc != 2) return 2;
    const auto dir = std::string(argv[1])+"/case-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    int failures = 0;
    auto check = [&](bool ok, const char* name) { std::printf("%s %s\n", ok?"PASS":"FAIL",name); failures += !ok; };
    auto sample = [](int32_t id) {
        PrefixFile f; f.identity="test-model"; f.tokens={10,20,id,40,50}; f.positions={2,5};
        f.blobs={PrefixBlob(256,(uint8_t)id),PrefixBlob(256,7)}; return f;
    };
    auto tokens = [](const PrefixFile& f) { std::vector<int64_t> v(f.tokens.begin(),f.tokens.end());v.push_back(99);return v; };
    auto a=sample(30), b=sample(31), c=sample(32); std::string error;
    {
        ConversationStore store;
        check(store.open(dir,"test-model",2,1600,error),"open private store");
        check(store.put(a,"",error),"save first full conversation branch");
        if (failures) return 1; // RED: the unimplemented store cannot save a branch.
        check(store.put(b,"",error),"save second branch sharing a system");
        check(store.contains(a.tokens) && store.contains(b.tokens) && !store.contains(c.tokens),
              "metadata dedup detects exact stored branches without loading state");
        check(!store.contains({10,20}), "dedup does not confuse a shorter prefix with a full snapshot");
        const auto bytes_before=store.bytes();
        check(store.put(a,"",error) && store.bytes()==bytes_before && store.size()==2,
              "duplicate put consumes no slots or additional bytes");
        const auto removed=store.find(tokens(a));
        fs::remove(removed.path);
        check(!store.contains(a.tokens), "deleted snapshot is not a metadata cache hit");
        check(store.put(a,"",error) && store.contains(a.tokens) && store.size()==2,
              "a deleted snapshot is rewritten without duplicating the index entry");
        const auto shortened=store.find(tokens(b));
        fs::resize_file(shortened.path, 1);
        check(!store.contains(b.tokens), "truncated snapshot is not a metadata cache hit");
        check(store.put(b,"",error) && store.contains(b.tokens) && store.bytes()==bytes_before,
              "a truncated snapshot is repaired with correct byte accounting");
        check(!store.open(dir,"test-model",2,1600,error) && store.size()==2,"reopen rejected without dropping lock or index");
        auto ha=store.find(tokens(a)), hb=store.find(tokens(b));
        check(ha.position==5 && hb.position==5 && ha.path!=hb.path,"A/B/A exact branch lookup");
        check(store.find(tokens(a),0,{ha.path}).position==2,"rejected candidate is excluded even if its file cannot be removed");
        check(store.find(tokens(a),0,{ha.path,hb.path}).position==0,"excluding every rejected candidate terminates at a miss");
        check(store.find({10,20,90,40,50,99}).position==2,"sibling only reuses common checkpoint");
        check(store.find({11,20,30,40,50,99}).position==0,"changed early token is a miss");
        check(store.find(tokens(a),5).position==0,"RAM wins on equal match");
        check(store.find({10,20,30,40,50}).position==2,"leave final token for decode");
        PrefixFile loaded;
        check(store.load(ha,loaded,error) && loaded.blobs==a.blobs,"restore exact state and touch LRU");
        check(store.put(c,"",error) && store.find(tokens(b)).position==2 && store.find(tokens(a)).position==5,"LRU evicts B, not touched A");
        check(store.evictions()==1,"budget eviction diagnostic counter");
        check(store.bytes()<=1600 && store.size()==2,"byte and slot bounds");
        auto huge=sample(35);huge.blobs={PrefixBlob(2000,1)};
        check(!store.put(huge,"",error) && store.size()==2,"oversize rejected without eviction");
        check(!store.put(b,"",error,[]{return true;}) && store.size()==2,"cancel before write preserves branches");
        auto hc=store.find(tokens(c));
        check(store.put(b,hc.path,error) && store.find(tokens(c)).position==5,"incoming restore protected from eviction");
        check(!store.load(hc,loaded,error,[]{return true;}) && store.find(tokens(c)).position==5,"cancel read does not discard valid cache");
        ConversationStore second;
        check(!second.open(dir,"test-model",2,1600,error),"exclusive directory lock");
        const auto perms=fs::status(dir).permissions();
        check((perms&(fs::perms::group_all|fs::perms::others_all))==fs::perms::none,"private directory");
        auto bad=store.find(tokens(b));
        std::fstream corrupt(bad.path,std::ios::binary|std::ios::in|std::ios::out);
        corrupt.seekp(-9,std::ios::end);corrupt.put('X');corrupt.close();
        check(!store.load(bad,loaded,error) && store.find(tokens(b)).position==2,"corrupt branch rejected and evicted");
        std::ofstream(dir+"/keep-user-file.txt")<<"untouched";
    }
    {
        ConversationStore store;
        check(store.open(dir,"test-model",2,1600,error) && store.find(tokens(c)).position==5,"restart indexes compatible branch");
        auto longer=c;longer.tokens.push_back(60);longer.positions.push_back(6);
        check(store.put(longer,"",error) && store.size()==1,"newer same branch supersedes shorter snapshot");
    }
    {
        ConversationStore store;
        check(store.open(dir,"different-model",2,1600,error) && store.size()==0,"identity change invalidates state");
        check(fs::exists(dir+"/keep-user-file.txt"),"cleanup never touches unrelated files");
    }
    fs::create_directory_symlink(dir,dir+"-link");
    ConversationStore symlink;
    check(!symlink.open(dir+"-link","test-model",2,1600,error),"symlink directory rejected");
    return failures?1:0;
}
