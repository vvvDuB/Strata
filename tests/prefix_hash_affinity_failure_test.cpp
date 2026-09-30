#include "strata/program/prefix_hash_batch.hpp"
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <filesystem>
#if defined(__linux__)
#include <pthread.h>
extern "C" int __real_pthread_setaffinity_np(pthread_t,size_t,const cpu_set_t*);
std::atomic_bool force_unavailable{false};
extern "C" int __wrap_pthread_setaffinity_np(pthread_t thread,size_t n,const cpu_set_t* mask) {
    return force_unavailable.load() ? EINVAL : __real_pthread_setaffinity_np(thread,n,mask);
}
#endif
using namespace strata::program;
int main(int argc,char**argv) {
#if !defined(__linux__)
    std::puts("SKIP Linux syscall failure test");return 0;
#else
    if(argc!=2)return 2;
    if(prefix_hash_worker_cpus().size()<2){std::puts("SKIP needs two allowed CPUs");return 0;}
    const auto directory=std::filesystem::path(argv[1])/std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    std::filesystem::create_directories(directory);
    PrefixFile file;file.identity="affinity-loss";file.tokens={1,2};file.positions={2};file.blobs.assign(8,PrefixBlob(2ull<<20,19));
    const auto caller=std::this_thread::get_id();bool wrong_caller=false;std::string error;int failures=0;
    auto check=[&](bool ok,const char* name){std::printf("%s %s\n",ok?"PASS":"FAIL",name);failures+=!ok;};
    // Simulate the OS denying a captured core after startup. Only the OS boundary
    // is wrapped; writer, reader, scalar checksums and join path are real.
    const auto existing=(directory/"preexisting-valid.bin").string();
    check(prefix_write(existing,file,error),"create independently valid fixture before OS affinity loss");
    force_unavailable.store(true);
    auto cancel=[&]{wrong_caller|=std::this_thread::get_id()!=caller;return false;};
    const auto path=(directory/"valid.bin").string();
    check(prefix_write(path,file,error,kPrefixMaxBytes,cancel),"unavailable worker CPU falls back before writing valid cache");
    PrefixFile loaded;
    check(prefix_read(existing,loaded,false,error,kPrefixMaxBytes,cancel) && loaded.blobs==file.blobs,
          "unavailable worker CPU never makes a valid snapshot look corrupt");
    check(!wrong_caller,"fallback cancellation callback stays on caller");
    std::vector<PrefixHashJob> jobs;for(auto&blob:file.blobs)jobs.push_back({blob.data(),blob.size()});
    std::vector<uint64_t> hashes{123};
    check(!prefix_hash_batch(jobs,hashes,error,[]{return true;}) && error=="cancelled" && hashes==std::vector<uint64_t>{123},
          "cancellation wins over CPU fallback without publishing hashes");
    force_unavailable.store(false);
    check(prefix_hash_batch(jobs,hashes,error) && hashes==std::vector<uint64_t>(jobs.size(),prefix_hash(file.blobs[0].data(),file.blobs[0].size())),
          "workers remain usable after the failed-affinity batch");
    return failures?1:0;
#endif
}
