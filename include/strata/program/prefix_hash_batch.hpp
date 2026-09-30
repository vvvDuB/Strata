#pragma once
#include "strata/program/prefix_file.hpp"
#include "strata/program/prefix_hash_affinity.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <thread>

namespace strata::program {
struct PrefixHashJob { const void* data; size_t size; };
inline size_t prefix_hash_worker_count(uint64_t bytes, size_t jobs, unsigned available) {
    if (!jobs) return 0;
    if (bytes < (8ull << 20) || available <= 1) return 1;
    return std::min({size_t(4), jobs, size_t(available)});
}
namespace prefix_hash_detail {
// Borrowed records and result storage must outlive every worker, on all exits.
struct JoinWorkers {
    std::atomic_bool& stop;
    std::vector<std::thread>& threads;
    ~JoinWorkers() {
        stop.store(true, std::memory_order_relaxed);
        for (auto& thread : threads) if (thread.joinable()) thread.join();
    }
};
inline bool hash_parallel(const std::vector<PrefixHashJob>& jobs, const std::vector<int>& cpus,
                          size_t workers, std::vector<uint64_t>& computed,
                          const std::function<void()>& check_cancelled) {
    std::atomic_bool affinity_failed{false};
    {
        std::atomic_bool stop{false};
        std::atomic_size_t next{0}, finished{0};
        std::vector<std::thread> threads;
        prefix_hash_detail::JoinWorkers join{stop, threads};
        threads.reserve(workers);
        for (size_t i = 0; i < workers; ++i) {
            threads.emplace_back([&, cpu = cpus[i]] {
                if (!prefix_hash_detail::pin_worker(cpu)) {
                    affinity_failed.store(true, std::memory_order_relaxed);
                    stop.store(true, std::memory_order_relaxed);
                    finished.fetch_add(1, std::memory_order_release);
                    return;
                }
                while (!stop.load(std::memory_order_relaxed)) {
                    const auto index = next.fetch_add(1, std::memory_order_relaxed);
                    if (index >= jobs.size()) break;
                    computed[index] = prefix_hash(jobs[index].data, jobs[index].size);
                }
                finished.fetch_add(1, std::memory_order_release);
            });
        }
        // Only the caller invokes potentially thread-affine cancellation.
        while (finished.load(std::memory_order_acquire) != workers) {
            check_cancelled();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    } // Join before any fallback or before publishing computed hashes.
    return !affinity_failed.load(std::memory_order_relaxed);
}
}
inline bool prefix_hash_batch(const std::vector<PrefixHashJob>& jobs, std::vector<uint64_t>& hashes,
                              std::string& error, const std::function<bool()>& cancelled = {}) {
    try {
        auto check_cancelled = [&] {
            if (cancelled && cancelled()) throw std::runtime_error("cancelled");
        };
        check_cancelled();
        uint64_t bytes = 0;
        for (const auto& job : jobs) {
            if (job.size && !job.data) throw std::runtime_error("missing checksum input");
            bytes += std::min(uint64_t(job.size), std::numeric_limits<uint64_t>::max() - bytes);
        }
        std::vector<uint64_t> computed(jobs.size());
        const auto cpus = prefix_hash_worker_cpus();
        const auto workers = prefix_hash_worker_count(bytes, jobs.size(), (unsigned)cpus.size());
        auto hash_on_caller = [&] {
            for (size_t i = 0; i < jobs.size(); ++i) {
                check_cancelled();
                computed[i] = prefix_hash(jobs[i].data, jobs[i].size);
            }
        };
        if (workers <= 1) hash_on_caller();
        else if (!prefix_hash_detail::hash_parallel(jobs, cpus, workers, computed, check_cancelled)) {
            // CPUs may become unavailable after startup. Join every worker and
            // recompute on the caller rather than report a valid file corrupt.
            hash_on_caller();
        }
        check_cancelled();
        hashes = std::move(computed);
        return true;
    } catch (const std::exception& e) {
        error = e.what();
        return false;
    }
}
} // namespace strata::program
