#include "strata/program/prefix_hash_batch.hpp"
#include <atomic>
#include <cstdio>
#include <thread>
using namespace strata::program;
int main() {
    int failures = 0;
    auto check = [&](bool ok, const char* name) {
        std::printf("%s %s\n", ok ? "PASS" : "FAIL", name); failures += !ok;
    };
    check(prefix_hash_worker_count(32ull<<20,64,16)==4,"large batch bounded to four workers");
    check(prefix_hash_worker_count(32ull<<20,2,64)==2,"no more workers than records");
    check(prefix_hash_worker_count(32ull<<20,64,2)==2,"respect available CPU bound");
    check(prefix_hash_worker_count(32ull<<20,64,0)==1,"unknown hardware safely falls back to caller");
    check(prefix_hash_worker_count(512ull<<10,64,16)==1,"small metadata remains synchronous");
    check(prefix_hash_worker_count(32ull<<20,0,16)==0,"empty batch does not launch threads");
    std::vector<PrefixBlob> input(8,PrefixBlob(4ull<<20));
    std::vector<PrefixHashJob> jobs;
    std::vector<uint64_t> expected;
    for (size_t i=0;i<input.size();++i) {
        for (size_t j=0;j<input[i].size();++j) input[i][j]=uint8_t(i*31+j*17+j/257);
        jobs.push_back({input[i].data(),input[i].size()});
        expected.push_back(prefix_hash(input[i].data(),input[i].size()));
    }
    jobs.push_back({nullptr,0}); expected.push_back(1469598103934665603ull);
    std::vector<uint64_t> hashes{99}; std::string error;
    check(prefix_hash_batch(jobs,hashes,error) && hashes==expected,"batch preserves every scalar FNV checksum and record order");
    check(prefix_hash_batch({{input[0].data(),2},{nullptr,0}},hashes,error) &&
          hashes==std::vector<uint64_t>{prefix_hash(input[0].data(),2),1469598103934665603ull},
          "scalar fallback and empty records preserve checksum semantics");
    const auto caller=std::this_thread::get_id(); std::atomic_bool wrong_thread{false};
    std::atomic_int polls{0}; hashes={123,456};
    check(!prefix_hash_batch(jobs,hashes,error,[&]{
        if (std::this_thread::get_id()!=caller) wrong_thread.store(true);
        return ++polls>2;
    }) && error=="cancelled" && hashes==std::vector<uint64_t>{123,456},
          "in-flight cancellation joins workers and does not publish partial checksums");
    check(!wrong_thread.load(),"cancellation callback stays on caller thread");
    polls=0;
    check(!prefix_hash_batch(jobs,hashes,error,[&]() -> bool {
        if(++polls>2)throw std::runtime_error("caller cancellation failure");
        return false;
    }) && error=="caller cancellation failure" && hashes==std::vector<uint64_t>{123,456},
          "throwing callback joins workers before returning unchanged outputs");
    check(!prefix_hash_batch({{nullptr,1}},hashes,error) && hashes==std::vector<uint64_t>{123,456},
          "missing nonempty input rejected before worker reads");
    check(!prefix_hash_batch({},hashes,error,[]{return true;}) && hashes==std::vector<uint64_t>{123,456},
          "cancelled empty batch does not publish an empty replacement");
    check(prefix_hash_batch({},hashes,error) && hashes.empty(),"empty successful batch publishes empty result");
    for (size_t i=0;i<input.size();++i)
        check(prefix_hash(input[i].data(),input[i].size())==expected[i],"borrowed input unchanged after success/cancellation");
    return failures ? 1 : 0;
}
