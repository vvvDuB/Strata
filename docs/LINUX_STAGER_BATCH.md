# Linux expert read batches

The batching portion of [upstream PR #1323](https://github.com/Niko1221/Strata/pull/1323)
(commit `b89e2aef42bbc46a5a9d30d82659719427df2518`) is ported onto this fork's
existing Linux O_DIRECT/AIO reader. It changes expert transfer during prefill,
including native GGUF role slices and experts.bin. NVFP4 encoding and the
KV grow/elastic cache coordinator are unchanged.

Each stager worker claims consecutive jobs and calls `copy_blobs` for the
file-backed jobs. The Linux reader merges touching 4 KiB-aligned windows into
requests of at most 32 MiB and submits a wave before waiting for completions.
The resident RAM copy is checked through `resident_blob`, which follows RAM/GPU
exchange rotations. Failed batched reads retry the individual reads and retain
the mapped fallback.

The default is eight jobs per batch for file-backed sources, one for other
sources. `STRATA_STAGER_BATCH` selects 1 through 32; it is also clamped to the
stager ring size. The ring's DMA events must complete before a worker overwrites
any claimed buffer. Cancellation still joins the issuer, finishes the stager
and drains the GPU streams before returning borrowed cache memory.

Linux retains the GGUF/file mappings. The PR's NTFS mapping-close optimization
is not included. No additional startup script or flag is needed: the existing
bash launcher runs the rebuilt binary and uses the batching default. For a
comparison with the old per-job scheduling, set `STRATA_STAGER_BATCH=1` in the
process environment.

## Validation

Measured on RTX 5070 12 GiB / SM120, Ryzen 5700X3D, 32 GiB RAM, CUDA 13.4,
IQ3_XXS, NVFP4 KV, context 131072, resident expert budget 22 GiB and VRAM
reserve 640 MiB. Prompt/conversation cache reuse was disabled during the
performance comparison. The candidate was rebuilt locally against the current KV/monitor changes.


Four primary sessions used old/new/new/old order, each with an 8K warm-up,
an uncached 8K request and an uncached 32K request. A separate candidate
batch-1 session checked the previous scheduling. One candidate session was
replaced with a fresh default-batch session because GPU sanitizer activity
overlapped its 32K request. That overlapped session remains in the raw results
with its measurements marked as excluded. No other timings were discarded.

| Uncached prefill | Previous binary, batch 1 | Linux batches, batch 8 |
| --- | ---: | ---: |
| 8K, first retained run | 1176.37 tok/s | 1198.59 tok/s |
| 8K, second retained run | 1085.39 tok/s | 1168.72 tok/s |
| 32K, first retained run | 208.29 tok/s | 1413.63 tok/s |
| 32K, second retained run | 1337.38 tok/s | 369.41 tok/s |

The mean 8K rate increased by 4.67% in this small sample. The 32K rates varied
strongly in both arms: the apparent mean gain of 15.36% does not establish a
stable long-context speedup. The candidate batch-1 control measured 1096.93
tok/s at 8K and 1343.90 at 32K, without a preceding warm-up.
All measured requests returned exactly the same token stream (`731` and the
turn terminator), read the full prompt with zero reuse, grew KV/cache slots
from 16384/3354 to 40960/3256, and returned to 8192/3354 after short arithmetic.
This checks transfer parity on these prompts, not general model quality.

The candidate's sampled peak RSS was 26.73 GiB, versus 26.06 GiB for the old
binary's session with usable memory samples. All sampled sessions reported
zero process swap. The first old session's memory sample summary was lost
when the initial harness treated the turn terminator as answer text and then
read a zombie /proc entry; its completed native request rates are retained.

The three core/GPU tests passed (file source, VMM and KV/cache elasticity),
plus the three resident rotation variants and eleven monitor/VRAM server
tests. File-source cases verify exact bytes, destination guards, empty and
invalid batches, mapped copies and direct reads, duplicated/out-of-order
experts and adjacent ranges spanning the 32 MiB merge limit. Mixed resident
and file copies are checked after repeated RAM/GPU exchanges. Compute
Sanitizer memcheck, run without the loaded model, reported zero errors.
A real odd-sized ring (five buffers, three workers, requested batch 32
clamped to five) completed two consecutive 8K prefills. A 64K request
was then cancelled and drained in 3.80 seconds; subsequent arithmetic
returned `4` and restored all 3354 expert slots.

[Raw rates and counters](../bench/results/2026-10-07-linux-stager-batch/validation.json)
record the native SHA256 hashes and excluded session. The production service
was left stopped. No release was published.
