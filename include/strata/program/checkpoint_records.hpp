#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace strata::program::checkpoint_records {
// Fork persistence payload v2: the indexer accumulator and position are state,
// not scratch. The outer PrefixFile envelope stays unchanged; its identity
// carries the payload version so three-buffer v1 records are never replayed.
inline constexpr size_t count = 5;
using Blobs = std::vector<std::vector<uint8_t>>;

template<class Checkpoint>
void append(Blobs& blobs, const Checkpoint& c) {
    blobs.push_back(c.gdn); blobs.push_back(c.ple); blobs.push_back(c.tails);
    blobs.push_back(c.dead); blobs.push_back(c.block_pos);
}

template<class Loans, class Checkpoint>
void borrow(Loans& loans, const Checkpoint& c) {
    loans.emplace_back(c.gdn); loans.emplace_back(c.ple); loans.emplace_back(c.tails);
    loans.emplace_back(c.dead); loans.emplace_back(c.block_pos);
}

inline bool valid(const Blobs& blobs, size_t records, const std::array<size_t, count>& sizes) {
    if (records > blobs.size() / count) return false;
    for (size_t i = 0; i < records; ++i)
        for (size_t j = 0; j < count; ++j)
            if (blobs[i * count + j].size() != sizes[j]) return false;
    return true;
}

template<class Checkpoint>
bool take(Blobs& blobs, size_t record, Checkpoint& c) {
    if (record >= blobs.size() / count) return false;
    const size_t i = record * count;
    c.gdn = std::move(blobs[i]); c.ple = std::move(blobs[i + 1]);
    c.tails = std::move(blobs[i + 2]); c.dead = std::move(blobs[i + 3]);
    c.block_pos = std::move(blobs[i + 4]);
    return true;
}
} // namespace strata::program::checkpoint_records
