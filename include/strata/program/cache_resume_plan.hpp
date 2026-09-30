#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
namespace strata::program::cache_resume {
// Candidates are metadata only. Choose first, validate/load the winner, then
// restore once. A rejected candidate is removed and selection runs again.
enum class Source { active, ram, legacy_disk, system_disk, shared_disk };
struct Candidate {
    Source source = Source::active;
    int64_t tokens = 0;
    size_t index = 0;
    uint64_t bytes = std::numeric_limits<uint64_t>::max();
};
inline int transfer_tier(Source source) {
    return source == Source::active ? 0 : source == Source::ram ? 1 : 2;
}
inline Candidate select(std::span<const Candidate> candidates) {
    Candidate best;
    for (const auto& c : candidates) {
        if (c.tokens <= 0) continue;
        const bool equal = c.tokens == best.tokens;
        const int tier = transfer_tier(c.source), old_tier = transfer_tier(best.source);
        if (c.tokens > best.tokens || (equal && (tier < old_tier ||
            (tier == old_tier && c.bytes < best.bytes)))) best = c;
    }
    return best;
}
// A formatting/retry tail costs less to replay than copying an entire session.
// Keep full images for real edits, deep truncation, steering/image changes and
// conversation switches. Never infer reuse from this heuristic: token/state
// compatibility and checkpoint validation remain authoritative.
inline bool should_preserve(int64_t previous_end, int64_t common, bool compatible) {
    if (previous_end <= 0) return false;
    if (!compatible) return true;
    common = std::clamp<int64_t>(common, 0, previous_end);
    return previous_end - common > 64;
}
inline bool final_checkpoint(int64_t end, int64_t latest, int64_t capacity) {
    return capacity > 1 && end > 0 && latest >= 0 && end > latest && end - latest > 64;
}
} // namespace strata::program::cache_resume
