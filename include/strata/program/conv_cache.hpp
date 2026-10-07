// include/strata/program/conv_cache.hpp - the conversation cache's retention policy: which checkpoint
// leaves when the cache is over its slot budget.
//
// The cache holds conversation checkpoints: snapshots of the running state (the GDN recurrence, the PLE
// history, the QSA indexer tails - `ConvCheckpoint` in generate.cpp) that a request resumes its prompt
// from instead of reading those tokens again.  The request path keeps only the checkpoints whose tokens
// are a prefix of the current prompt - any other checkpoint's positional cells have been overwritten,
// because the KV cache is one arena that holds one branch of history at a time - so at any moment the
// retained checkpoints are a CHAIN: sorted by length, each a prefix of the next.  That is a radix cache's
// tree collapsed onto the one branch of history the session can hold.
//
// The chain's root is therefore the deepest point every request so far has shared - in practice the end
// of the system prompt, which every new chat of the same client mounts through - and it is exactly the
// node a radix cache keeps alive while its leaves rotate.  First-in-first-out kept it only by accident of
// being first: after `prompt_cache` newer checkpoints it was gone, and the next chat read the whole
// prefix again at prefill speed.  So:
//
// `balanced` (default) retains sparse coverage across the branch, bounded system checkpoints,
// the latest leaf and one learned fork point. `lru` preserves the older root-pinned rotation.
// Both policies stay within --prompt-cache slots and require exact token/image prefixes.
// Checkpoints contain recurrent state, not independent KV branches. A rewritten suffix is
// always recomputed; retention only reduces the replay of its unchanged common prefix.
#pragma once

#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <vector>
#include <limits>

namespace strata::program::conv_cache {

// Persisted stamps belong to an earlier process's clock. Keep relative age and
// ties, but remove arbitrary absolute values before advancing the local clock.
// Allocate before mutating; callers may decline promotion on allocation failure.
template<class Checkpoint>
void rebase_stamps(std::vector<Checkpoint>& checkpoints) {
    std::vector<uint64_t> stamps;
    stamps.reserve(checkpoints.size());
    for (const auto& c : checkpoints) stamps.push_back(c.used);
    std::sort(stamps.begin(), stamps.end());
    stamps.erase(std::unique(stamps.begin(), stamps.end()), stamps.end());
    for (auto& c : checkpoints)
        c.used = uint64_t(std::lower_bound(stamps.begin(), stamps.end(), c.used) - stamps.begin()) + 1;
}

/// The index in `stamps` of the chain item to drop once the chain holds more than `cap` items.  `stamps`
/// are the items' last-use stamps; the caller owns the chain and erases the returned index.  Pure and
/// deterministic so conv_cache_test.cpp can walk the scenarios by hand.
inline size_t eviction_victim(const uint64_t* stamps, size_t n, int64_t cap, size_t prefix_count = 1) {
    if (cap < 2 || n < 2) return 0;   // no room for root and leaf: the pin is off, the oldest leaves
    // Protect intermediate system checkpoints, but always leave space for recent conversation state.
    // With small legacy caps, preserve the original root + leaf behaviour.
    const size_t pins = std::min(n - 1, std::min(std::max<size_t>(1, prefix_count),
                                              (size_t) std::max<int64_t>(1, cap - 2)));
    size_t v = pins;
    for (size_t i = pins + 1; i < n; ++i)
        if (stamps[i] < stamps[v]) v = i;
    return v;
}

// A single positional branch can contain many checkpoints. Compare its token prefix once,
// not once per checkpoint. Tokens remain the authority; no fuzzy or hash-only reuse.
template<class A, class B>
inline int64_t common_prefix(const std::vector<A>& cached, const std::vector<B>& prompt) {
    size_t i = 0, end = std::min(cached.size(), prompt.size());
    while (i < end && cached[i] == prompt[i]) ++i;
    return static_cast<int64_t>(i);
}

// Lengths are strictly increasing and describe ONE valid branch. Unlike LRU, minimise
// lost coverage: removing p_i adds (p_i-p_{i-1})*(p_{i+1}-p_i) to the total replay work
// for uniformly distributed fork positions. Keep the newest point, the bounded system
// pins, and (when there is room) the most recently observed exact fork boundary.
// Ties keep established checkpoints rather than repeatedly shifting the sparse grid.
inline size_t balanced_eviction_victim(const std::vector<int64_t>& positions, int64_t cap,
                                      int64_t system_end = 0, int64_t anchor = 0) {
    const size_t n = positions.size();
    if (cap < 2 || n < 2) return 0;
    size_t prefix_count = 0;
    while (prefix_count < n && positions[prefix_count] <= system_end) ++prefix_count;
    const size_t pins = std::min(n - 1, std::min(std::max<size_t>(1, prefix_count),
                                              static_cast<size_t>(std::max<int64_t>(1, cap - 2))));
    std::vector<bool> keep(n, false);
    for (size_t i = 0; i < pins; ++i) keep[i] = true;
    // A very long system can exceed the pin budget. Keep its actual end, not just
    // the first few periodic points; otherwise unrelated chats lose the full root.
    if (prefix_count > pins && positions[prefix_count - 1] == system_end) {
        keep[pins - 1] = false;
        keep[prefix_count - 1] = true;
    }
    keep[n - 1] = true;
    size_t kept = static_cast<size_t>(std::count(keep.begin(), keep.end(), true));
    if (kept < static_cast<size_t>(cap) && anchor > 0)
        for (size_t i = 0; i < n; ++i)
            if (positions[i] == anchor) { keep[i] = true; break; }

    size_t victim = n;
    long double best = std::numeric_limits<long double>::infinity();
    for (size_t i = 0; i + 1 < n; ++i) {
        if (keep[i]) continue;
        const int64_t left = i ? positions[i - 1] : 0;
        const long double loss = static_cast<long double>(positions[i] - left) *
                                 static_cast<long double>(positions[i + 1] - positions[i]);
        if (loss <= best) { best = loss; victim = i; }
    }
    // For a valid over-budget chain at least one item is unpinned.
    return victim == n ? 0 : victim;
}

inline std::vector<int64_t> read_boundaries(int64_t from, int64_t end, int64_t root,
                                          int64_t turn, int64_t reread, int64_t every,
                                          int64_t fork = -1) {
    std::vector<int64_t> out;
    for (const int64_t to : {reread, root, turn, fork, end})
        if (to > from && to <= end) out.push_back(to);
    // Checkpoints require complete recurrent state, unavailable inside a batched chunk. Split the
    // read at exact boundaries rather than rounding --prompt-cache-every up to the prefill size.
    if (every > 0 && from >= 0 && end > from) {
        const int64_t step = every - from % every;
        if (step < end - from)
            for (int64_t to = from + step; to < end;) {
                out.push_back(to);
                if (every >= end - to) break;
                to += every;
            }
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

/// --prompt-cache-tail: the extra checkpoint near the prompt's end (`tail[i]`) is the first to go - it serves
/// only a branch of the last request - unless it is the newest item (the one just saved: `n > 1` and the newest
/// stamp).  With no such item this is the plain policy.  Never the root.
inline size_t eviction_victim(const uint64_t* stamps, size_t n, int64_t cap, const bool* tail) {
    if (tail != nullptr && n >= 2 && cap >= 2) {
        size_t newest = 0;
        for (size_t i = 1; i < n; ++i)
            if (stamps[i] > stamps[newest]) newest = i;
        for (size_t i = 1; i < n; ++i)
            if (tail[i] && i != newest) return i;
    }
    return eviction_victim(stamps, n, cap);
}

}  // namespace strata::program::conv_cache
