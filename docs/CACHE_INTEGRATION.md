# Fork-compatible conversation-cache integration

## Provenance and scope

This integration starts at fork `f330adfa16cf54ff9824d47b25b08fb0440cecf8`
and merges the shared snapshot/RAM and eviction-only disk changes from:

- Niko1221/Strata#189, `f1b0c1647899259010879c99a035af999e78e0e2`.
- Niko1221/Strata#190, `18c475e4b6dfea517a47f34ed95c427ed12a7901`.
  Its history already includes #189; it is merged once, not duplicated.
- The stream-ordered pinned-transfer idea from Niko1221/Strata#203,
  `3b6100c101fbcee6612144854991a51eb1eb481b`, adapted rather than cherry-picked.

The original authorship of #189/#190 remains in the merge history. The #190
source retains credit to Marmaduke Woodman's NVMe work and the admission and
identity reviewers. The optional pinned-transfer adaptation is based on
q8atnight's #203. Existing MIT licenses remain intact.

## Existing launch behavior remains the default

The fork's balanced retention, exact interval/root/turn/learned-fork boundaries,
checkpoint-buffer recycling, prefix coverage diagnostics, HTTP/FIFO/epoch STOP
handling, cached-token accounting, direct Python CLI and Anthropic effort
mapping remain in place. The CPU changes from fork PR #2 and GPU kernels are
not changed by this integration.

In particular, **the old options are not reinterpreted as a RAM cache**:

| Options | Meaning |
| --- | --- |
| `--prefix-cache-file FILE` | Existing fork system-prefix persistence |
| `--conversation-cache-dir DIR` | Existing fork multi-conversation disk store |
| `--conversation-cache-mib N` | Existing disk-store quota, default 8192 MiB |
| `--conversation-cache-slots N` | Existing disk-store slots, default 4 |
| `--prompt-cache-policy balanced\|lru` | Existing checkpoint retention policy |

No new cache is enabled without its own flags. An unchanged launch command
continues using the original storage backend. The old backend's single-GPU,
resident INT8, MTP, prompt-root and no-vision/control-vector restrictions remain.

The checkpoint representation now also retains `idx_dead` and `idx_block_pos`.
The fork's persistence payload uses five state blobs instead of three and has
an explicit **v2 identity**. It never guesses or replays a v1 payload as v2.
A new executable/configuration already invalidates legacy cache identity; expect
an initial cold request. The legacy store's existing startup cleanup removes
incompatible entries. **Use a separate cache directory when testing a new
binary so the active installation's snapshots are not touched.** No automatic
conversion of existing snapshots is attempted.

## New shared RAM and disk tiers

Upstream RAM controls have a separate namespace in this fork:

```text
--conversation-ram-cache-mib N           default 0 (off)
--conversation-ram-cache-slots N         default 4
--conversation-ram-cache-min-free-mib N  default 2560
```

These park the used main/draft KV, recurrence, PLE, indexer state and retained
checkpoints on a branch switch or rewind. They choose the longest compatible
prefix without changing execution into concurrent model sessions. Unchanged
K/V pages can be retained only after a real restore and are invalidated on any
rewrite or legacy disk restore. Admission counts retained capacity and held
incoming images as well as the physical-memory floor.

The optional #190 disk tier requires `-DSTRATA_ENABLE_CONVERSATION_DISK=ON`
and OpenSSL Crypto, plus enabled RAM and prompt caching:

```text
--conversation-cache-disk DIR
--conversation-cache-disk-mib N          default 0 (off)
--conversation-cache-disk-slots N        default 128
--conversation-cache-tokenizer DIR
--conversation-cache-template FILE
```

The wrapper supplies its actual tokenizer and template identity. This store
uses a distinct `strata-conversations-v1` subdirectory, SHA-256 asset/runtime
identity, file integrity checks, exclusive ownership and bounded atomic writes.
It writes **on RAM eviction**, not on every response; the most recent active
turn is not guaranteed to survive an engine restart. Cold asset hashing can be
expensive. The two disk implementations are not wire-compatible.

The legacy store, live branch, checkpoints and new tiers can coexist. The
outgoing branch is captured before a restore/reset can overwrite its cells;
legacy branch/system entries are applied only when their prefix is longer.
A full shared snapshot is validated before GPU writes. Transfer failure remains
fatal instead of continuing from a half-restored session. Cache I/O cancellation
uses the existing request ticket, removes incomplete temporary writes, and
never publishes a partially read image. Already issued GPU transfers must drain
before cancellation can return safely.

Keep shared parking disabled for multi-GPU; upstream #189 does not support
whole-session multi-GPU parking. Existing multi-GPU exact-boundary operation
without parking stays synchronous. New RAM/disk budgets are additional to the
model, old checkpoint and legacy-store budgets, not an automatic redistribution
of them. No machine-specific budget is chosen here.

## #203: opt-in bounded staging, not a speed claim

`STRATA_CK_ASYNC=1` enables pinned D2H staging **only at periodic, exact
single-GPU boundaries**. `STRATA_CK_SYNC=1` takes precedence. Root, turn and
learned-fork checkpoints, multi-GPU operation, and the default path remain
synchronous.

One pending capture owns a checkpoint slot and recycled payload buffers. It
copies all five state fields on the producing stream, records an event, and
publishes only after event completion. It is drained before the next checkpoint,
end-of-prefill/cancellation/DONE, and on every exceptional teardown. Allocation
failure before transfer falls back to the synchronous capture. Transfer/event
failure cannot publish a checkpoint. The extra pinned staging allocation is
one running-state image (roughly the size of a checkpoint), outside the RAM
conversation-cache quota.

Unlike the original PR, no background vector-copy worker is added: the existing
exact-boundary scheduler and one serving thread own publication. The author's
latest 0.1.28 A/B on #203 found no measurable prompt-wall-time gain (72.20 s
async versus 72.16 s sync median on a single RTX 3090). Therefore this feature
is off by default and **has no asserted inference speedup**.

## Repeatable host validation

```sh
cmake -S . -B build-cache-host -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=OFF \
  -DSTRATA_NATIVE_EXPERTS=OFF -DSTRATA_BUILD_TESTS=OFF \
  -DSTRATA_BUILD_CONVERSATION_TESTS=ON -DSTRATA_ENABLE_CONVERSATION_DISK=ON
cmake --build build-cache-host -j2 --target \
  conversation_cache_test conversation_memory_test conversation_file_test \
  shared_conversation_store_test conversation_io_cancel_test \
  checkpoint_records_test checkpoint_staging_test conv_cache_test \
  conv_cache_policy_test request_stop_test prefix_file_test conversation_store_test
ctest --test-dir build-cache-host --output-on-failure
python3 -m unittest discover -s serve -p 'test_*.py'
python3 -m unittest discover -s tests -p 'test_*.py'
python3 -m unittest discover -s tools -p 'test_conversation_cache_*.py'
python3 -O -m unittest discover -s tools -p 'test_conversation_cache_*.py'
python3 -m unittest discover -s tools -p 'test_cache_fork_integration.py'
```

The original `cmake -S tests` standalone suites also remain available. Re-run
host CTests with `-fsanitize=address,undefined -fno-omit-frame-pointer` in a
separate debug build to exercise malformed payloads and injected failures.
Replay helpers remove both new and legacy cache paths from supplied configs;
only their private output directories should receive test snapshots.

## Hardware validation still required

Host tests, mocked stream/transfer failure tests and CUDA compilation do not
establish correct full-model restore, real DMA ordering, cancellation latency,
or inference speed on the Ryzen 7 5700X3D/RTX 5070. Before deployment, compare
the exact fork baseline against this branch using identical IQ3_XXS assets and
settings, a new private cache namespace, and the upstream parity, isolation,
growth, disk-restart and HTTP-smoke gates under `tools/`.

Exercise: unchanged launch flags; repeat/append/edit/learned fork; A/B/A return;
RAM pressure; eviction/restart; corrupt and foreign images; STOP during prefill
and disk I/O; sync versus opt-in staging. Keep correctness fingerprints separate
from timing runs. Measure reused tokens, replay gap, TTFT and ms/round; never
translate a host codec result into an end-to-end tok/s claim. Windows/HIP and
multi-GPU runtime validation are not claimed by this integration.
