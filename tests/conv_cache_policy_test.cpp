// CPU-only regression/simulator. Exercises the SAME pure policy used by generate.cpp.
// Token counts are measured policy outcomes, not GPU throughput or real-model parity.
#include "strata/program/conv_cache.hpp"
#include <algorithm>
#include <cstdio>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>
using namespace strata::program::conv_cache;
namespace {
int failures = 0;
void check(bool ok, const char* name) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", name); failures += !ok;
}
struct Chain {
    std::vector<int64_t> positions;
    std::vector<uint64_t> stamps;
    uint64_t clock = 0;
    int cap = 12;
    int64_t root = 4540, anchor = 0;
    bool balanced = true;
    void save(int64_t p) {
        if (cap <= 0 || p < 1) return;
        auto same = std::find(positions.begin(), positions.end(), p);
        if (same != positions.end()) { stamps[same - positions.begin()] = ++clock; return; }
        positions.push_back(p); stamps.push_back(++clock);
        if (positions.size() <= static_cast<size_t>(cap)) return;
        size_t pins = static_cast<size_t>(std::count_if(positions.begin(), positions.end(),
                                                       [&](int64_t x) { return x <= root; }));
        size_t v = balanced ? balanced_eviction_victim(positions, cap, root, anchor)
                            : eviction_victim(stamps.data(), stamps.size(), cap, pins);
        positions.erase(positions.begin() + v); stamps.erase(stamps.begin() + v);
    }
    int64_t resume(int64_t common) const {
        auto it = std::upper_bound(positions.begin(), positions.end(), common);
        return it == positions.begin() ? 0 : *--it;
    }
    void prune(int64_t resume) {
        auto end = std::upper_bound(positions.begin(), positions.end(), resume);
        stamps.resize(end - positions.begin()); positions.erase(end, positions.end());
    }
    void read(int64_t from, int64_t end, int64_t fork = -1, int64_t every = 2048) {
        for (auto p : read_boundaries(from, end, root, end, -1, every, fork)) save(p);
    }
};
uint64_t step(uint64_t h, int32_t token) { return (h ^ static_cast<uint32_t>(token)) * 1099511628211ull; }
int benchmark() {
    std::puts("{\"kind\":\"CPU retention simulation; not GPU performance\",\"scenarios\":[");
    bool first = true;
    for (const int64_t end : {42421, 65531, 131067}) {
        Chain old, now; old.balanced = false;
        old.read(0, end); now.read(0, end);
        uint64_t old_gap = 0, new_gap = 0;
        int64_t old_worst = 0, new_worst = 0;
        for (int64_t p = 4540; p <= end; ++p) {
            const auto a = p - old.resume(p), b = p - now.resume(p);
            old_gap += a; new_gap += b;
            old_worst = std::max(old_worst, a); new_worst = std::max(new_worst, b);
        }
        if (!first) std::puts(",");
        first = false;
        const int64_t split = 16046;
        std::printf("{\"prompt_checkpoint\":%lld,\"slots\":12,\"system_end\":4540,"
                    "\"fork_at\":16046,\"lru_resume\":%lld,\"balanced_resume\":%lld,"
                    "\"lru_mean_replay_gap\":%.3f,\"balanced_mean_replay_gap\":%.3f,"
                    "\"lru_worst_replay_gap\":%lld,\"balanced_worst_replay_gap\":%lld,"
                    "\"balanced_positions\":[", (long long)end, (long long)old.resume(split),
                    (long long)now.resume(split), double(old_gap)/(end-4540+1),
                    double(new_gap)/(end-4540+1), (long long)old_worst, (long long)new_worst);
        for (size_t i=0;i<now.positions.size();++i)
            std::printf("%s%lld", i ? "," : "", (long long)now.positions[i]);
        std::printf("]}");
    }
    std::puts("\n]}"); return 0;
}
}
int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--benchmark") return benchmark();
    check(common_prefix(std::vector<int32_t>{1,2,3}, std::vector<int64_t>{1,2,4}) == 2,
          "exact mixed-width token comparison stops at first changed token");
    check(common_prefix(std::vector<int32_t>{}, std::vector<int64_t>{1}) == 0,
          "empty prefix has no reusable tokens");
    check(common_prefix(std::vector<int32_t>{1,2,3}, std::vector<int64_t>{1,2}) == 2,
          "truncated prompt reports the exact common prefix");
    check(read_boundaries(12288, 42612, 4540, 42612, -1, 2048, 16046).at(1) == 16046,
          "learned divergence gets an exact non-grid checkpoint");
    const auto max = std::numeric_limits<int64_t>::max();
    check(read_boundaries(max-3, max, -1, -1, -1, max) == std::vector<int64_t>{max},
          "periodic boundary arithmetic does not overflow");
    {
        Chain old, now; old.balanced = false;
        old.read(0, 42421); now.read(0, 42421);
        check(old.resume(16046) == 4540, "reproduce root-plus-recent-tail LRU hole at 16k");
        check(now.resume(16046) == 14336, "balanced policy retains 14336/16046 unchanged tokens");
        now.prune(now.resume(16046)); now.anchor = 16046;
        now.read(now.positions.back(), 42612, 16046);
        check(now.resume(16046) == 16046, "learned fork survives the remaining 26k prefill");
        for (int64_t end=44660; end<131000; end+=2048) now.save(end);
        check(now.resume(16046) == 16046, "learned fork survives subsequent append-only growth");
        check(now.positions.size() == 12, "long contexts stay within the original twelve-slot budget");
    }
    {
        Chain c; c.root = 65531; c.read(0, 100000);
        check(std::find(c.positions.begin(), c.positions.end(), c.root) != c.positions.end(),
              "large system keeps its actual end when early prefix pins exhaust their budget");
        check(c.positions.back() == 100000, "newest conversation checkpoint is always retained");
    }
    {
        bool bounded = true, latest = true, ordered = true, deterministic = true;
        std::mt19937 rng(75321);
        for (int cap=1; cap<=32; ++cap) {
            Chain c; c.cap=cap; c.root=4540;
            int64_t p=0;
            for (int round=0; round<400; ++round) {
                p += 1 + rng()%4096;
                c.save(p);
                bounded &= c.positions.size() <= static_cast<size_t>(cap);
                latest &= c.positions.back() == p;
                ordered &= std::adjacent_find(c.positions.begin(), c.positions.end(),
                                               std::greater_equal<int64_t>()) == c.positions.end();
                auto positions=c.positions; positions.push_back(p+1);
                deterministic &= balanced_eviction_victim(positions,cap,c.root,c.anchor) ==
                                 balanced_eviction_victim(positions,cap,c.root,c.anchor);
                if (cap>3 && !c.positions.empty() && round%13==0) c.anchor=c.positions[c.positions.size()/2];
            }
        }
        check(bounded && latest && ordered && deterministic,
              "12800 seeded insertions: bounds, latest point, sorted chain, deterministic policy");
    }
    {
        // Toy recurrent state + positional cells: rollback at a retained checkpoint and replay
        // must agree with a complete reread, including edits, truncations and insertions.
        bool parity = true, safe = true;
        std::mt19937 rng(8341);
        for (int cap : {1,2,3,6,12}) {
            Chain c; c.cap=cap; c.root=64;
            std::vector<int32_t> previous, cells;
            std::vector<uint64_t> saved(8192, 0);
            for (int round=0; round<250; ++round) {
                std::vector<int32_t> prompt=previous;
                if (prompt.empty()) prompt.assign(257,42);
                switch (round%4) {
                    case 0: prompt.insert(prompt.end(), 1+rng()%200, int32_t(rng()%1000)); break;
                    case 1: prompt[rng()%prompt.size()] = int32_t(1000+rng()%1000); break;
                    case 2: if (prompt.size()>10) prompt.resize(1+rng()%(prompt.size()-1)); break;
                    default: prompt.insert(prompt.begin()+rng()%prompt.size(), 15, int32_t(rng()%1000)); break;
                }
                auto common=common_prefix(previous,prompt);
                int64_t resume=c.resume(std::min<int64_t>(common,prompt.size()-1));
                safe &= resume<=common && resume<(int64_t)prompt.size();
                c.prune(resume);
                c.anchor = common>0 && common<(int64_t)previous.size() ? std::min<int64_t>(common,prompt.size()-1) : 0;
                const int64_t fork=c.anchor>resume ? c.anchor : -1;
                const auto boundaries=read_boundaries(resume,prompt.size(),c.root,prompt.size(),-1,64,fork);
                uint64_t state=resume ? saved[resume] : 1469598103934665603ull;
                cells.resize(prompt.size());
                size_t b=0;
                for (size_t i=resume;i<prompt.size();++i) {
                    cells[i]=prompt[i]; state=step(state,prompt[i]);
                    if (b<boundaries.size() && (int64_t)i+1==boundaries[b]) {
                        saved[i+1]=state; c.save(i+1); ++b;
                    }
                }
                uint64_t full=1469598103934665603ull;
                for (auto t:prompt) full=step(full,t);
                parity &= full==state && cells==prompt;
                previous=prompt;
            }
        }
        check(safe && parity, "1250 seeded append/edit/truncate/insert replays preserve toy recurrent and positional state");
    }
    return failures ? 1 : 0;
}
