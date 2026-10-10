# Upstream 0.1.42 integration

This fork merges the official `v0.1.42` release, commit
`61b3fb5dd3f1e8ec09cf7e4e05208bc6d3c46406`, into the owner branch based on
`5a2100deeaec6c6c991a4761c926df20d39af2d4`. The owner's uncommitted Pi
compatibility changes were saved as a separate parent commit before the merge.

The version is **0.1.42**, intended as a stable release. Publication is a
separate owner decision after validation.

## Retained owner features

- NVFP4 KV, its RHT codec, native conversion, GPU unpack and attention paths.
- Elastic KV growth and the segmented expert cache, including live monitor
  statistics and temporary on-demand vision leases.
- Profile-ranked RAM experts and the bounded RAM LRU, optional elastic page
  reclamation, synchronized statistics and Linux batched expert reads.
- Balanced checkpoints with exact root, turn and learned fork boundaries,
  buffer reuse and cache timing and cached-token reporting.
- RAM conversation parking and optional disk/prefix persistence, including
  staging admission, identity validation and spill callbacks.
- HTTP disconnect/queue cancellation, request epochs and per-layer prefill
  cancellation. Queued requests check cancellation every 100 ms; heartbeat
  delivery keeps the upstream interval.
- Direct server CLI, numeric reasoning budgets, Anthropic effort mapping and
  automatic detection of Pi's standalone summarization requests. Pi summaries
  disable thinking and receive compact-checkpoint guidance without a new flag
  or feature-specific console message. Ordinary requests keep their policy.
- On-demand `--once` vision encoding with disk-cached embeddings and GPU cache
  release/refill; resident vision and upstream lazy loading remain available.

## Integration decisions

Upstream CUDA kernels, pinned stage buffers and DMA completion fences are
retained. The elastic RAM LRU does not release a buffer with an outstanding DMA
read. Reallocated slots retain matching pointer-index and event bookkeeping.

The owner batched `copy_blobs` implementation remains the common path, including
LRU hits. With a private RAM LRU enabled, automatic Linux buffered/direct I/O
selection keeps the owner's full-working-set memory admission. The upstream
skew-based estimate is used without that LRU. An explicit I/O override still
takes precedence.

Upstream shared-prefix pins, minimum parked-conversation length and cgroup-aware
RAM admission are combined with owner spill/staging behavior. Pinned entries
are not evicted to make room. The existing owner `--conversation-ram-cache-*`
names retain their RAM meaning; legacy `--conversation-cache-mib` and
`--conversation-cache-slots` retain their disk-store meaning.

Upstream tail skipping, measured PCIe fraction, eligible CPU prefill sharing,
I/O prefetch controls, parallel-serving code and platform fixes are included.
The upstream default `STRATA_ROUTE_TAIL_SKIP=7` can slightly change generated
text; `STRATA_ROUTE_TAIL_SKIP=0` disables it. No new speedup measurement is
claimed for this integration.

The legacy v1 disk codec retains its existing restrictions: Linux, single GPU,
resident INT8 KV, MTP and no vision/control vector. Enabling the optional disk
build does not relax those restrictions or turn persistence on by default.

## Validation

Validation hardware: Ryzen 5700X3D, 32 GiB RAM, RTX 5070 / SM120, Debian Linux,
CUDA 13.4. The native engine was built with optional conversation-disk support;
the production profile continues to use RAM parking and NVFP4 KV.

- Serving Python: 767 tests run, 8 skipped, no failures.
- Tooling Python: 709 tests run, 5 skipped, no failures.
- Request-capture Python: 1 test passed.
- Standalone host policy/snapshot suite: 16 tests passed.
- GPU/expert suite: 7 tests passed (expert source, elastic RAM LRU, elastic VRAM,
  NVFP4 GPU codec, native conversion, unpack and elastic KV).
- Optional conversation-cache suite: 8 tests passed (RAM policy, split failure,
  memory admission, cancellation, both file codecs, shared store and reclaim).
- Native PLE batched parity and the owner's native GGUF layout/source test
  passed as standalone executables.

Real-model smoke tests used the native SC117 abliteration IQ3_S weights,
NVFP4 KV with growth, a 22 GiB total RAM expert budget (18 GiB profile + 4 GiB
LRU), 12 balanced checkpoints and on-demand NVFP4-MIXED vision projection.
Eleven requests passed: cold/repeated/extended text, forced function calling,
length-cap reporting, numeric thinking budget, automatic Pi summarization,
46,838-token prefill and repeat, image recognition and text after the image.
The repeat reported 46,831 cached input tokens. The Pi summary completed with
1,101 output tokens and no reasoning content, below its 2,048-token limit.
These synthetic checks establish behavior, not model quality or speed gains.

A separate streaming disconnect test observed the monitor update from 8,192 to
98,304 KV cells and from 2,558 to 2,211 GPU expert slots. Cancellation stopped
prefill, a replacement request succeeded in 3.995 seconds, and the KV/cache
returned to their short-context allocation.
The separately rebuilt vision binary also passed three requests: a new green
image (cold encoder), the same image again (cached embedding) and text after the
encoder exited. The temporary lease restored all prior expert slots.

No HIP, SYCL, Windows, multi-GPU or full-context runtime validation is claimed.
The known intermittent NVMe throughput drops are not resolved by this merge.
