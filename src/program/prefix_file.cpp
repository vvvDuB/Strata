#include "strata/program/prefix_file.hpp"
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <cstdio>
#if !defined(_WIN32)
#include <fcntl.h>
#include <unistd.h>
#endif
namespace strata::program {
bool prefix_is_system(const std::vector<int64_t>& tokens) {
    // v1 is deliberately limited to this Qwen tokenizer's <|im_start|>system\n.
    // The RAM root can be the first user turn when no system is rendered; never
    // persist that turn. Unknown/custom role encodings conservatively opt out.
    return tokens.size() >= 3 && tokens[0] == 248045 && tokens[1] == 8678 && tokens[2] == 198;
}
uint64_t prefix_hash(const void* p, size_t n) {
    uint64_t h=1469598103934665603ull;
    const auto* b=static_cast<const uint8_t*>(p);
    for(size_t i=0;i<n;++i) {h^=b[i];h*=1099511628211ull;}
    return h;
}
namespace {
constexpr char magic[]="STRAPFX1", footer[]="ENDPFX01";
constexpr uint64_t max_record=512ull<<20;
void validate_meta(const PrefixFile& f) {
    if(f.identity.empty() || f.identity.size()>65536 || f.tokens.empty() || f.tokens.size()>262144 ||
       f.positions.empty() || f.positions.size()>64 || f.positions.back()!=f.tokens.size())
        throw std::runtime_error("invalid prefix metadata");
    uint64_t prev=0;
    for(auto p:f.positions) {if(p<=prev || p>f.tokens.size()) throw std::runtime_error("invalid checkpoint positions");prev=p;}
}
template<class T> PrefixBlob bytes(const std::vector<T>& v) {
    PrefixBlob b(v.size()*sizeof(T));if(!b.empty())std::memcpy(b.data(),v.data(),b.size());return b;
}
template<class T> std::vector<T> values(const PrefixBlob& b) {
    if(b.size()%sizeof(T))throw std::runtime_error("misaligned metadata");
    std::vector<T> v(b.size()/sizeof(T));if(!b.empty())std::memcpy(v.data(),b.data(),b.size());return v;
}
}
bool prefix_write(const std::string& path,const PrefixFile& file,std::string& error,
                  uint64_t max_bytes,const std::function<bool()>& cancelled) {
#if defined(_WIN32)
    error="persistent prefix v1 currently requires Linux";return false;
#else
    std::string temp;FILE* out=nullptr;
    try {
        validate_meta(file);
        if(file.blobs.size()>256)throw std::runtime_error("too many state records");
        if(std::filesystem::exists(path))throw std::runtime_error("cache already exists");
        auto parent=std::filesystem::path(path).parent_path();
        if(!parent.empty() && !std::filesystem::exists(parent)) {
            std::filesystem::create_directories(parent);
            std::filesystem::permissions(parent,std::filesystem::perms::owner_all);
        }
        temp=path+".tmp.XXXXXX";
        int fd=mkstemp(temp.data());
        if(fd<0)throw std::runtime_error("cannot create private cache temporary file");
        out=fdopen(fd,"wb");if(!out){close(fd);throw std::runtime_error("fdopen failed");}
        uint64_t total=0;
        auto put=[&](const void* p,size_t n){
            if(cancelled && cancelled())throw std::runtime_error("cancelled");
            if(total>max_bytes || n>max_bytes-total || (n && std::fwrite(p,1,n,out)!=n))throw std::runtime_error("cache write failed or exceeds byte limit");
            total+=n;
        };
        auto record=[&](const void* p,size_t n){
            if(n>max_record)throw std::runtime_error("state record too large");
            uint64_t len=n,hash=prefix_hash(p,n);put(&len,8);put(&hash,8);put(p,n);
        };
        put(magic,8);record(file.identity.data(),file.identity.size());
        record(file.tokens.data(),file.tokens.size()*4);record(file.positions.data(),file.positions.size()*8);
        uint64_t count=file.blobs.size();put(&count,8);
        for(const auto& b:file.blobs)record(b.data(),b.size());
        put(footer,8);
        if(fflush(out) || fsync(fileno(out)))throw std::runtime_error("cache sync failed");
        fclose(out);out=nullptr;
        if(cancelled && cancelled())throw std::runtime_error("cancelled");
        // link publishes atomically and refuses to replace an existing cache, including a symlink.
        if(link(temp.c_str(),path.c_str()))throw std::runtime_error("cannot publish cache (already exists or I/O error)");
        unlink(temp.c_str());
        return true;
    } catch(const std::exception& e) {
        if(out)fclose(out);
        if(!temp.empty())unlink(temp.c_str());
        error=e.what();return false;
    }
#endif
}
bool prefix_read(const std::string& path,PrefixFile& file,bool metadata_only,std::string& error,
                 uint64_t max_bytes,const std::function<bool()>& cancelled) {
    try {
        if(!std::filesystem::is_regular_file(std::filesystem::symlink_status(path)) ||
           std::filesystem::file_size(path)>max_bytes)throw std::runtime_error("invalid cache file/size");
        std::ifstream in(path,std::ios::binary);if(!in)throw std::runtime_error("cannot open cache");
        uint64_t remaining=std::filesystem::file_size(path);
        auto get=[&](void* p,size_t n){
            if(cancelled && cancelled())throw std::runtime_error("cancelled");
            if(n>remaining || (n && !in.read(static_cast<char*>(p),(std::streamsize)n)))throw std::runtime_error("truncated cache");
            remaining-=n;
        };
        auto record=[&](uint64_t limit){
            uint64_t len,hash;get(&len,8);get(&hash,8);
            if(len>limit || len>remaining)throw std::runtime_error("invalid record length");
            PrefixBlob b((size_t)len);get(b.data(),b.size());
            if(prefix_hash(b.data(),b.size())!=hash)throw std::runtime_error("cache checksum mismatch");
            return b;
        };
        char m[8];get(m,8);if(std::memcmp(m,magic,8))throw std::runtime_error("unsupported cache format");
        PrefixFile f;
        auto id=record(65536);f.identity.assign(id.begin(),id.end());
        f.tokens=values<int32_t>(record(262144*4));f.positions=values<uint64_t>(record(64*8));validate_meta(f);
        if(!metadata_only) {
            uint64_t n;get(&n,8);if(n>256)throw std::runtime_error("too many state records");
            for(uint64_t i=0;i<n;++i)f.blobs.push_back(record(max_record));
            get(m,8);if(std::memcmp(m,footer,8) || remaining)throw std::runtime_error("invalid cache footer");
        }
        file=std::move(f);return true;
    } catch(const std::exception& e){error=e.what();return false;}
}
int64_t prefix_match(const PrefixFile& f,const std::vector<int64_t>& tokens) {
    size_t common=0;
    while(common<f.tokens.size() && common<tokens.size() && f.tokens[common]==tokens[common])++common;
    int64_t best=0;
    for(auto p:f.positions)if(p<=common && p<tokens.size())best=(int64_t)p;
    return best;
}
}
