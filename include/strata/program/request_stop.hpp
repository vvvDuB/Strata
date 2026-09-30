#pragma once
#include <atomic>
#include <cstdint>

namespace strata::program {
// Capture a ticket in the stdin reader when GEN arrives, not when inference starts.
// A later STOP cancels it even if GEN is still queued; an earlier STOP is stale.
class RequestStop {
    std::atomic<uint64_t> epoch_{0};
public:
    uint64_t ticket() const { return epoch_.load(); }
    void signal() { epoch_.fetch_add(1); }
    bool requested(uint64_t ticket) const { return epoch_.load() != ticket; }
};
}
