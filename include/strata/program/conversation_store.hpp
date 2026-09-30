#pragma once
#include "strata/program/prefix_file.hpp"
#include <functional>

namespace strata::program {
inline constexpr uint64_t kConversationMaxBytes = 4ull << 30;
struct ConversationHit { std::string path; int64_t position = 0; };
// Single writer, bounded disk backing for inactive text-only KV branches.
class ConversationStore {
public:
    ConversationStore() = default;
    ~ConversationStore();
    ConversationStore(const ConversationStore&) = delete;
    ConversationStore& operator=(const ConversationStore&) = delete;
    bool open(const std::string& dir, const std::string& identity, size_t slots,
              uint64_t budget, std::string& error);
    ConversationHit find(const std::vector<int64_t>& tokens, int64_t better_than = 0,
                         const std::vector<std::string>& excluded = {}) const;
    // Metadata-only deduplication before GPU capture and multi-GiB staging.
    bool contains(const std::vector<int32_t>& tokens) const;
    bool load(const ConversationHit&, PrefixFile&, std::string& error,
              const std::function<bool()>& cancelled = {});
    bool put(const PrefixFile&, const std::string& protected_path, std::string& error,
             const std::function<bool()>& cancelled = {}, const BorrowedPrefixBlobs& borrowed = {});
    void discard(const std::string& path);
    size_t size() const { return entries_.size(); }
    uint64_t bytes() const;
    uint64_t evictions() const { return evicted_; }
private:
    struct Entry { std::string path; PrefixFile meta; uint64_t bytes = 0, used = 0; };
    std::vector<Entry> entries_;
    std::string dir_, identity_;
    size_t slots_ = 0;
    uint64_t budget_ = 0, clock_ = 0;
    uint64_t evicted_ = 0;
    int lock_ = -1;
};
}
