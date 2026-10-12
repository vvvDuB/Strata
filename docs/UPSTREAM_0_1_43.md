# Upstream 0.1.43.1 integration

Owner stable version: **0.1.43**. This fork merges upstream `v0.1.43.1`,
commit `a0116b2782a5b82622358c04a06496485d12b2d6`, into owner main
`b9c5204d9be5b7f4d5f3e5d8e144b762697b09e5`.

The owner features listed in [UPSTREAM_0_1_42.md](UPSTREAM_0_1_42.md) remain:
NVFP4 KV, elastic KV and segmented expert cache, exact checkpoints, RAM and
optional disk conversation caches, automatic Pi summarization, request-epoch
cancellation and on-demand vision with temporary GPU memory leases.

## Upstream fixes

Adaptive overlap remains off by default, as in upstream 0.1.43.1. CUDA callback
warm-up and the pinned-memory completion signal are included. Direct PLE reads
are no longer switched automatically to mmap; mapped PLE read-ahead and the
longer disk activity allowance are retained. Non-finite output is retried once
before any token is emitted, or ends an already started reply as length. A
second failure before output returns a retryable server error. Failed attempts
are excluded from cached-input accounting.

Image preparation uses the request's existing disconnect event while waiting
for both the on-demand vision gate and the engine FIFO. It does not create a
second watcher or overwrite request-epoch cancellation.

## Resident RAM maintenance

`STRATA_RAM_ADAPT=N` is opt-in and unset by default. Already staged, frequently
used experts can replace cold same-layer RAM experts without additional file
reads or growing the private LRU. The outgoing expert remains in its staging
slot. Current routed experts are excluded from eviction, and a live DMA read
causes refusal rather than a blocking wait. The no-private-LRU path operates
after the layer's CPU expert jobs join. Concurrent prefetch, lookahead,
asynchronous GPU adaptation, adaptive overlap, pipeline windows and split,
remote or batch configurations are excluded where required.

On Linux, optional `STRATA_RAM_ADAPT_FILE_DROP=1` advises that full interior
clean file pages of newly promoted experts can be discarded; shared boundary
pages are preserved. It changes no weights and issues no file writes. This
retains the deployed conservative cold-victim policy; replacing used RAM
victims is not included. Diagnostic trace records remain file-only.

The private-LRU direct-I/O policy now survives the post-residency recheck.
Without a private LRU, the upstream partial file-cache policy remains available.

## Validation

Results for this integration are recorded in the release's `VALIDATION.json`.
The target machine is Debian Linux, Ryzen 5700X3D, 32 GiB RAM, RTX 5070 / SM120,
CUDA 13.4. HIP, SYCL, Windows and multiple GPUs are not validated on this host.
No improvement in minimum decode throughput or cure for intermittent NVMe
stalls is claimed. Release publication requires the owner's approval.
