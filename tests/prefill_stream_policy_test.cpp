// Process-wide startup policy only: no CUDA allocation or model loading.
#include "strata/prefill/prefill.hpp"
#include <cstdio>
#include <cstdlib>

int main() {
    using strata::prefill::Prefill;
    const char* value = std::getenv("STRATA_PREFILL_STREAM_MIN");
    const int64_t legacy = value ? std::atoll(value) : 2048;
    int failures = 0;
    auto check = [&](bool ok, const char* name) {
        std::printf("%s %s\n", ok ? "PASS" : "FAIL", name);
        failures += !ok;
    };
    check(Prefill::stream_min() == legacy, "absent CLI retains the legacy env/default");
    Prefill::set_stream_min(4096);
    check(Prefill::stream_min() == 4096, "explicit 4096 wins over the benchmark environment");
    Prefill::set_stream_min(2048);
    check(Prefill::stream_min() == 2048, "another positive CLI value remains supported");
    Prefill::set_stream_min(0);
    check(Prefill::stream_min() == legacy, "reset restores legacy resolution");
    return failures ? 1 : 0;
}
