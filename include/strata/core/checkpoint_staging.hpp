#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {
struct CheckpointCopy {
    const void* source = nullptr;
    void* destination = nullptr;
    size_t bytes = 0;
};
enum class CheckpointEnqueue { queued, unavailable, failed };

// One pending, bounded capture. Transport orders D2H copies on the producer's
// stream. Destinations must remain alive until finish/destruction; no snapshot
// is published before finish succeeds. No worker thread or extra checkpoint.
// Transport: reserve(bytes), enqueue(dst,src,bytes), record(), wait(), drain(),
// release(), data(). Every fallible operation receives an error string.
template<class Transport>
class CheckpointStaging {
    Transport& io_;
    std::vector<CheckpointCopy> copies_;
    bool pending_ = false;
public:
    explicit CheckpointStaging(Transport& io) : io_(io) {}
    CheckpointStaging(const CheckpointStaging&) = delete;
    CheckpointStaging& operator=(const CheckpointStaging&) = delete;
    ~CheckpointStaging() { if (pending_) io_.drain(); io_.release(); }
    bool pending() const noexcept { return pending_; }

    CheckpointEnqueue enqueue(std::vector<CheckpointCopy> copies, std::string& error) {
        if (pending_) { error = "checkpoint capture already pending"; return CheckpointEnqueue::failed; }
        size_t bytes = 0;
        for (const auto& c : copies) {
            if ((c.bytes && (!c.source || !c.destination)) ||
                c.bytes > std::numeric_limits<size_t>::max() - bytes) {
                error = "invalid checkpoint copy span"; return CheckpointEnqueue::failed;
            }
            bytes += c.bytes;
        }
        if (!bytes) { error = "empty checkpoint capture"; return CheckpointEnqueue::unavailable; }
        if (!io_.reserve(bytes, error)) return CheckpointEnqueue::unavailable;
        copies_ = std::move(copies);
        // Set before enqueueing: any later failure must drain in-flight DMA.
        pending_ = true;
        size_t offset = 0;
        for (const auto& c : copies_) {
            if (c.bytes && !io_.enqueue(io_.data() + offset, c.source, c.bytes, error)) {
                io_.drain(); pending_ = false; copies_.clear(); return CheckpointEnqueue::failed;
            }
            offset += c.bytes;
        }
        if (!io_.record(error)) {
            io_.drain(); pending_ = false; copies_.clear(); return CheckpointEnqueue::failed;
        }
        return CheckpointEnqueue::queued;
    }
    bool finish(std::string& error) {
        if (!pending_) return true;
        if (!io_.wait(error)) {
            io_.drain(); pending_ = false; copies_.clear(); return false;
        }
        size_t offset = 0;
        for (const auto& c : copies_) {
            if (c.bytes) std::memcpy(c.destination, io_.data() + offset, c.bytes);
            offset += c.bytes;
        }
        pending_ = false; copies_.clear(); return true;
    }
};
} // namespace strata::core
