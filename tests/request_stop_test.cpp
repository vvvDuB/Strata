#include "strata/program/request_stop.hpp"
#include <cassert>

int main() {
    strata::program::RequestStop stop;
    const auto first = stop.ticket(); // captured when GEN enters the input queue
    assert(!stop.requested(first));
    stop.signal();                   // STOP before the main thread dequeues GEN
    assert(stop.requested(first));   // starting GEN must not clear this STOP
    const auto second = stop.ticket();
    assert(!stop.requested(second)); // the old STOP is stale for the next GEN
    stop.signal();
    assert(stop.requested(second));  // STOP during prefill
    const auto third = stop.ticket();
    assert(!stop.requested(third));
}
