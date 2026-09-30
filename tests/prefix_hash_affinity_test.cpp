#include "strata/program/prefix_hash_batch.hpp"
#include <cstdio>
#if defined(__linux__)
#include <sched.h>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <unistd.h>
#endif
using namespace strata::program;
int main() {
#if !defined(__linux__)
    std::puts("SKIP Linux affinity regression"); return 0;
#else
    cpu_set_t original; CPU_ZERO(&original);
    if (sched_getaffinity(0,sizeof original,&original)) return 2;
    struct Restore { cpu_set_t cpus; ~Restore(){sched_setaffinity(0,sizeof cpus,&cpus);} } restore{original};
    std::vector<int> allowed;
    for (int i=0;i<CPU_SETSIZE;++i) if(CPU_ISSET(i,&original))allowed.push_back(i);
    if(allowed.size()<3){std::puts("SKIP needs three permitted CPUs");return 0;}
    std::vector<int> workers(allowed.begin()+1,allowed.begin()+std::min(allowed.size(),size_t(5)));
    prefix_hash_configure_worker_cpus(workers); // capture BEFORE host restriction
    cpu_set_t host; CPU_ZERO(&host); CPU_SET(allowed[0],&host);
    if(sched_setaffinity(0,sizeof host,&host))return 2;
    std::vector<PrefixBlob> data(16,PrefixBlob(16ull<<20,73));
    std::vector<PrefixHashJob> jobs;for(auto& b:data)jobs.push_back({b.data(),b.size()});
    const auto scalar=prefix_hash(data[0].data(),data[0].size());
    std::set<std::string> existing;for(auto& p:std::filesystem::directory_iterator("/proc/self/task"))existing.insert(p.path().filename().string());
    std::set<int> observed;bool escaped=false;int extra=0;
    auto observe=[&]{
        for(auto& p:std::filesystem::directory_iterator("/proc/self/task")) {
            const auto tid=p.path().filename().string();if(existing.count(tid))continue;
            ++extra;
            // Do not test the transient inherited mask before the new worker
            // has initialized. CPU time proves it has begun hashing.
            std::ifstream stat(p.path()/"stat");std::string line;std::getline(stat,line);
            if(line.empty())continue;
            std::istringstream fields(line.substr(line.rfind(')')+2));std::string value;
            for(int index=0;index<=11;++index)fields>>value; // field14:utime
            if(value.empty() || std::stoull(value)==0)continue;
            cpu_set_t mask;CPU_ZERO(&mask);if(sched_getaffinity(std::stoi(tid),sizeof mask,&mask))continue;
            if(CPU_COUNT(&mask)!=1)escaped=true;
            for(int cpu=0;cpu<CPU_SETSIZE;++cpu) if(CPU_ISSET(cpu,&mask)) {
                observed.insert(cpu);
                if(std::find(workers.begin(),workers.end(),cpu)==workers.end())escaped=true;
            }
        }
        return false;
    };
    std::vector<uint64_t> hashes;std::string error;int failures=0;
    auto check=[&](bool ok,const char* name){std::printf("%s %s\n",ok?"PASS":"FAIL",name);failures+=!ok;};
    check(prefix_hash_batch(jobs,hashes,error,observe) && hashes==std::vector<uint64_t>(jobs.size(),scalar),"affinity-aware batch preserves scalar checksums");
    check(!escaped && observed==std::set<int>(workers.begin(),workers.end()),"workers use distinct captured permitted CPUs despite pinned host");
    cpu_set_t after;CPU_ZERO(&after);sched_getaffinity(0,sizeof after,&after);
    check(CPU_EQUAL(&host,&after),"checksum work never changes caller affinity");
    // Explicitly unavailable workers must stay synchronous, not escape the
    // one-CPU caller via hardware_concurrency(). Also covers unconfigured mode.
    prefix_hash_configure_worker_cpus({});extra=0;
    check(prefix_hash_batch(jobs,hashes,error,observe) && extra==0,"empty worker plan uses caller without launching inherited-mask workers");
    sched_setaffinity(0,sizeof original,&original);
    prefix_hash_configure_worker_cpus({workers[0],workers[0]});extra=0;
    sched_setaffinity(0,sizeof host,&host);
    check(prefix_hash_batch(jobs,hashes,error,observe) && extra==0,"one unique CPU stays synchronous even with duplicate plan entries");
    return failures?1:0;
#endif
}
