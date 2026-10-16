# Upstream 0.1.44.1 integration

Owner version and stable release title: **0.1.44**. This integrates upstream
`v0.1.44.1`, commit `296b59d2f73c8f04a805df29c2fa97655adace78`, into owner
`0.1.43`, commit `5f0c76fa881b302a360c8d921646d11a8ac2d2db`.
Publication remains a separate owner decision after validation.

## Retained owner behavior

The features in [UPSTREAM_0_1_43.md](UPSTREAM_0_1_43.md) remain: NVFP4 KV and
its RHT codec, elastic KV/VRAM, segmented expert caches, exact root/turn/fork
checkpoints and asynchronous staging, RAM and optional disk conversation
caches, automatic Pi summarization, request-epoch cancellation, direct server
arguments and on-demand vision with temporary GPU memory leases.

Conservative resident-RAM adaptation and promoted-file-page advice stay opt-in.
The private-LRU I/O recheck is retained. No model weights are converted or
changed by this integration, and the installed model selection is preserved.
Upstream NVFP4 experts are a separate format from the owner's NVFP4 KV cache.
Their setup tests and the owner's KV setup tests live in separate modules.

The previous owner merge omitted the decode dispatch's routing-counter
allocation. This integration restores it for GPU adaptation and opt-in RAM
adaptation, including RAM adaptation with GPU swaps frozen. Runtime validation
checks actual RAM promotion trace records, not only expert byte transfers.

## Merge decisions

- The detached stdin reader owns its shared state, as upstream now requires;
  cancellation retains the owner's request tickets and signal method. Queued
  STOP messages cannot cancel the next unrelated request, and the reader does
  not refer to stack-local mutexes after QUIT.
- Upstream prompt-direct expert reads are combined with owner batched reads,
  staging traces, RAM maintenance and pointer/DMA bookkeeping.
- Owner checkpoint buffer preparation is kept separate from checkpoint saving.
  Metal's running-state capture remains in the save operation.
- Exact checkpoint selection and validation across owner cache backends remain.
  Upstream optional prompt-head aliasing and image-embedding mismatch reporting
  are combined with them; aliasing remains off unless explicitly selected.
- The upstream elastic resident image encoder and owner on-demand leases retain
  their respective lifetime policies. Shared cancellation reaches image
  preparation and the upstream image-generation endpoint.
- Upstream engine setup refactoring, model-family defaults, Responses reasoning
  defaults and startup crash diagnostics are retained.

## Validation scope

Tests and runtime results are recorded in the release's `VALIDATION.json`.
The target is Debian Linux, Ryzen 5700X3D, 32 GiB RAM, RTX 5070 / SM120 and
CUDA 13.4. The published native CPU build is specific to this host's ISA.
HIP, SYCL, Metal, Windows and multiple GPUs are not validated on this host.
Other upstream model families and image generation are not tested with their
weights here. No minimum decode-throughput improvement or cure for the known
intermittent NVMe stalls is claimed.
