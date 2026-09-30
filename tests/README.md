# PLE batched-prefill regression

`native_ple_batch_test.cpp` calls the actual CUDA batched postops and compares
its output and final history against sequential single-token postops. It needs
no model weights. Build from the repository root with CUDA enabled:

```sh
cmake --build build --target native_ple_batch_test -j4
for t in 1 2 8 9 10 64 65 77 256 1024; do
  compute-sanitizer --tool memcheck --error-exitcode 99 build/native_ple_batch_test "$t" || exit $?
done
```

CUDA 12.8 / SM 120: the original unrolled convolution loop produces invalid
global reads at T=77; `#pragma unroll 1` removes them. The tested batch sizes
have zero sanitizer errors and zero absolute difference from sequential
postops. This does not claim every compiler/architecture has the same issue.

For the opt-in whole-model regression, start the Coder server with prefill 1024,
context 4096, MMQ enabled and the normal short-read threshold 64 (NOT 4096):

```sh
STRATA_TEST_URL=http://127.0.0.1:11435 \
STRATA_TEST_RESULTS=/tmp/strata-prefill-results \
python3 tools/test_prefill_regression.py -v
```

It tests the original 79-token coding crash, a prompt larger than one batch,
and a smaller prompt after relayout. The server is not started or stopped by
this test. Ordinary unittest discovery skips the live tests unless opted in.

## Native mmap experts

`local_file_expert_source_test` uses small synthetic files with real native blob
formats, no GPU work and no model weights. It checks canonical compatibility,
variable per-layer sizes/offsets, expert-index bounds, strict file-size checks,
reopening, isolation from a subsequently loaded global layout, and full unmap
(on Linux, including the last page).

```sh
cmake --build build --target local_file_expert_source_test -j4
build/local_file_expert_source_test /tmp/strata-mmap-test-new
```

The supplied fixture directory must not already exist; fixtures are retained
for inspection. The current mmap path requires `iq_pack.py --experts-bin`.


## Conversation cache retention (2026-09-29)

CPU-only tests can be built independently of the incomplete upstream suite:

```sh
cmake -S tests -B /tmp/strata-cache-tests
cmake --build /tmp/strata-cache-tests -j4
ctest --test-dir /tmp/strata-cache-tests --output-on-failure
/tmp/strata-cache-tests/conv_cache_policy_test --benchmark
python3 -m unittest discover -s serve -p 'test*.py'
```

`conv_cache_policy_test` invokes the same eviction policy used by the engine.
It checks coverage, learned forks, system/leaf retention, slot bounds and replay
in a toy recurrent state machine. Its JSON token counts are **CPU retention
simulation**, not GPU throughput or real-model numerical parity.

`tools/benchmark_prompt_cache.py` is an opt-in HTTP benchmark against an already
running caller-owned server. Run `--help` for options. It reports actual API cache
counts and observed TTFT, including repeat, append, history edit and branch switch.
It writes no conversation text. Its SSE parser has offline tests in
`serve/test_prompt_cache_benchmark.py`.

See `../../Docs/Strata-cache-optimization-2026-09-29.md` from the engine root for
implementation details, limitations and build instructions. Measurements are in
`bench/results/2026-09-29-cache-coverage/`.
