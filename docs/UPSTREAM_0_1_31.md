# Official 0.1.31 with owner serving/cache patches

Integration parents: owner `dfacbce34ba5f7df93e740d1866e0752518c295b` and
upstream tag `v0.1.31` (`9259cad4cfa3543cd3b8decab5962672b968c649`).
This is a full source/server merge, not the earlier native-only prototype.

## Retained owner behavior

- Balanced exact-token checkpoint coverage, bounded root pinning, learned fork
  boundaries, 12 checkpoints/4096 in the existing Q3 Bash deployment.
- V6 exact candidate ranking and invalid-candidate fallback, immutable synchronous
  checkpoint loans, bounded parallel checksum workers with CPU-affinity fallback,
  cache phase clocks, and the legacy cache writeback stall fix.
- HTTP disconnect/queued cancellation, request STOP epochs, per-layer prefill
  cancellation and draining, FIFO-owned status/history finalization.
- Direct shell-only wrapper options, correct OpenAI/Anthropic cache counters,
  built-in per-request Claude effort mapping and CUDA 12.8 PLE workaround.
- AVX2 IQ4_NL scale reuse and all existing expert-source validation.

No test service, autostart, JSON model profile, extra reasoning flag, dedicated
Claude system-prefix flag or V7 preventive checkpoint policy is introduced.
The optional RAM/shared-NVMe tiers remain opt-in and off in the Q3 deployment.

## 0.1.31 integration decisions

The official native kernels, expert-layout/sharded GGUF support, coupled drafting,
layer-carved sessions, tokenizer/frontend updates, tool truncation handling,
load/unload and idle unload support, GPU headroom/preload hooks, graceful SIGTERM,
and optional thinking-budget continuation are retained. New optional settings
keep their upstream defaults; no launcher configuration is required.

The **expert streaming threshold defaults to 2048**, as chosen by the owner after
the Q3/32 GiB trial. Upstream 1024 caused a large small-prefill regression on this
host. `STRATA_PREFILL_STREAM_MIN` remains an explicit benchmark override. This
threshold is separate from `--prefill 4096` and checkpoint interval 4096.

DONE retains the owner positional coverage fields and phase extensions; expert
RAM/file metrics use keyed extensions. The Python reader also accepts official
0.1.31 positional expert-tier metrics and older DONE layouts. Thinking-budget
continuations aggregate clocks and expert counters across engine segments, but
report only the **initial client prompt** cache hit. They never count internal
continuation tokens as cached client input.

Core snapshot sizes and restore validation use the session's allocated layer
carve, not the whole-model geometry. Owner staging/preallocation and incremental
snapshot APIs remain available. Synchronous exact-boundary checkpointing joins
all pipeline stages before capture; actual multi-GPU performance is not claimed.

The optional shared-NVMe envelope is now **STRSNAP version 2**, including
`layer_lo`/`layer_hi` and a v2 asset identity namespace. Its v1 images safely miss;
its durable fsync contract is unchanged. The production **legacy conversation
store** remains independent and retains its existing codec/writeback behavior.
Build/config identity can invalidate prior legacy snapshots: preserve the old
files but expect a first cold request after rebuilding. Never publish snapshots,
private logs, model files, authentication keys or core dumps.

## Validation and limitations

See `bench/results/2026-10-01-v0131-owner-integration/summary.json` for final
counts-only evidence. Validation uses only the installed IQ3_XXS weights for
inference, one RTX 5070, CUDA 12.8 / SM120, resident INT8 KV, mmap experts and MTP.
Synthetic layout/kernel tests are not tests of other model weights.

Standalone cache suites run normally and under ASan/UBSan. Server tests cover
both APIs, cancellation, status handover, effort, thinking budgets and usage.
GPU snapshot/PLE and IQ3 constituent-format bitwise kernel checks are separate
from the live Q3 conversation tests. New parser/default/carve/budget-statistic
regressions were observed failing before the corresponding integration fixes.

No full 128k input, real multi-GPU, HIP, idle-unload-under-production-load, or
uniform performance improvement is claimed. The earlier ABBA trial at streaming
2048 removed the observed small-prefill regression; warm decode was essentially
unchanged and cold-prefill variability remains. Changes to an earlier prompt
still invalidate its suffix: caching requires an exact positional token prefix.
Expert cache hit percentage is not the prompt-cache reuse percentage.
