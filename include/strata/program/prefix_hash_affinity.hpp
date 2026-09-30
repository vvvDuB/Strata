#pragma once
#include <algorithm>
#include <mutex>
#include <vector>
#include <utility>
#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace strata::program {
namespace prefix_hash_detail {
inline std::vector<int> current_cpus() {
    std::vector<int> cpus;
#if defined(__linux__)
    cpu_set_t allowed; CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof allowed, &allowed) == 0)
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
            if (CPU_ISSET(cpu, &allowed)) cpus.push_back(cpu);
#endif
    return cpus;
}
struct CpuPlan {
    std::mutex mutex;
    bool configured = false;
    std::vector<int> cpus;
};
inline CpuPlan& cpu_plan() { static CpuPlan plan; return plan; }
inline bool pin_worker(int cpu) {
#if defined(__linux__)
    if (cpu < 0 || cpu >= CPU_SETSIZE) return false;
    cpu_set_t selected; CPU_ZERO(&selected); CPU_SET(cpu, &selected);
    return pthread_setaffinity_np(pthread_self(), sizeof selected, &selected) == 0;
#else
    (void)cpu; return false;
#endif
}
}
// Configure once at engine startup BEFORE SessionLoopScratch pins the host.
// The engine supplies its existing physical-core list, excluding the host core.
// Never broaden the startup caller's/taskset mask; explicit empty means scalar.
inline void prefix_hash_configure_worker_cpus(const std::vector<int>& cpus) {
    const auto allowed = prefix_hash_detail::current_cpus();
    std::vector<int> selected;
    for (const int cpu : cpus)
        if (std::find(allowed.begin(), allowed.end(), cpu) != allowed.end() &&
            std::find(selected.begin(), selected.end(), cpu) == selected.end()) selected.push_back(cpu);
    auto& plan = prefix_hash_detail::cpu_plan();
    std::lock_guard<std::mutex> lock(plan.mutex);
    plan.cpus = std::move(selected);
    plan.configured = true;
}
inline std::vector<int> prefix_hash_worker_cpus() {
    auto& plan = prefix_hash_detail::cpu_plan();
    std::lock_guard<std::mutex> lock(plan.mutex);
    return plan.configured ? plan.cpus : prefix_hash_detail::current_cpus();
}
} // namespace strata::program
