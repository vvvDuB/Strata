# Exact cache resume and snapshot-transfer optimization (V6)

This integration retains the owner's v0.1.29 serving/cache/CUDA and AVX2 changes.
It does not enable the optional conversation RAM/shared-NVMe tiers, change model
profiles, increase checkpoint slots, or add a systemd service.

## Included

- Rank active state, checkpoints, RAM, shared disk, legacy disk and system prefix
  metadata globally before performing a GPU restore. Invalid candidates are
  excluded for the current request; a safe fallback is retried without loops.
- Avoid parking full images for compatible small (at most 64-token) same-chain
  rewinds/retries. Real branch switches and larger rewrites retain persistence.
- Capture a selective final prefill checkpoint for raw prompts with a substantial
  gap, while preserving balanced root/leaf/learned-fork retention and slot limits.
- Loan immutable completed recurrent checkpoint buffers to the synchronous legacy
  writer instead of making another full staging copy. Loans are not retained.
- Verify the unchanged FNV record checksums with bounded parallel workers, maximum
  four, for sufficiently large records. Capture the permitted physical CPU list
  before the session pins the host; checksum workers do not inherit its single
  core. CPU restrictions, empty/one-core masks and OS pin failures fall back to
  scalar verification. Workers are joined before success, cancellation or failure.
- Report optional phase timing fields after the fixed DONE protocol fields, with
  backward-compatible Python parsing. Timings distinguish selection, parking,
  read, restore, checkpoint, unchanged/new/mixed prefill and first generated token.

All reuse remains exact-token/image/state based, on one positional KV branch.
Changing a suffix still requires recomputation. The final prefill token belongs
in verify/decode; diagnostic replay excludes that token. Timings are host wall
clock, not CUDA events or a guarantee of first visible HTTP output latency.

## Deliberately excluded

The V7 preventive current-input checkpoint was not promoted. It reduced replay
but did not demonstrate a repeatable end-to-end latency benefit and increased
under-cap snapshot storage. No new input-marker retention hint is included here.
This integration is not a fix for the intermittent CUDA/watchdog request stall.
The subsequently diagnosed legacy disk-barrier stall and its separate fix are
documented in [CACHE_WRITEBACK.md](CACHE_WRITEBACK.md); V6 alone did not fix it.

## Validation scope

The isolated V6 Q3 IQ3_XXS 40k ABBA transfer test completed 40/40 known-answer
requests. Conversation returns improved by 37.5%, 38.3% and 42.9% in that workload,
with the same reused-token counts. These are two observations per case, not a
universal throughput guarantee. Cold/append timings and outliers were retained.
A preceding recurrent-loan test reduced staging memory; it did not establish a
universal latency improvement.

CPU cache/codec/planning/cancellation/timing/affinity suites, sanitizers and Python
serving tests cover the integration. Separate Q3 tests verified warm retry and
legacy/shared-disk restores, queued/disconnect cancellation and usage reporting.
Warm state fingerprints are not proof of universal cold-vs-warm numerical/logit
parity. No full 128k input, live-agent quality matrix or multi-GPU/HIP performance
claim is made. The intermittent watchdog stall remains an open limitation.

Standalone CPU suites:

```bash
cmake -S tests -B /tmp/strata-cache-tests -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/strata-cache-tests -j3
ctest --test-dir /tmp/strata-cache-tests --output-on-failure
python3 -m unittest discover -s serve -p 'test_*.py'
```

The binary/build identity invalidates older persistent cache images. Keep the
existing launcher/cache namespaces; a first request after rebuild may be cold.
No dedicated Claude system-prefix flag is introduced or re-enabled.
