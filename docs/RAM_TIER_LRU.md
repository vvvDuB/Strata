# RAM expert LRU

The LRU portion of [upstream PR #1324](https://github.com/Niko1221/Strata/pull/1324)
is ported from `20e55bc454d33b408f1f3a2c12b72aed5e66e45d`, after its #1323 dependency.
This fork already has the Linux batched reader. The port preserves its native GGUF
role reads, resident exchange rotation, NVFP4 KV and KV/cache elasticity.

`--resident-budget-gib N --resident-lru-gib L` gives `N-L` GiB to the resident
experts selected by the profile and reserves L GiB for recently read file experts.
L must be positive, finite and smaller than N. Without the LRU option the previous
retention policy remains selected. The stage pool can temporarily exceed its
retention target when every buffer is still protected by active work.

Decode reads admit entries and refresh their recency. The prompt path can copy an
entry already held by the pool, including in `copy_blobs`, but neither admits nor
refreshes it. A prompt that sweeps cold experts therefore does not replace the
entries reused by decode. The existing recent-layer/sequence protection remains
in force while pointers are used by expert computation.

The installed launcher selects 22 GiB total, with 18 GiB by profile and a fixed
4 GiB LRU, through one added native argument after `--`. No extra launcher is
required. The total is a budget for expert retention, not a process RSS limit:
model weights, CUDA mappings, prefill staging and other allocations also use RAM.

## Elastic mode and tracing

`STRATA_LRU_KEEP_FREE_GIB=G` enables the upstream watcher. Every 500 ms it reduces
the retention cap and releases cold, unprotected buffers when available memory
falls below G. It permits growth in steps of up to 1 GiB when availability exceeds
G by 2 GiB. The minimum cap is 1 GiB, rounded to whole expert buffers. Protected
buffers can delay reclamation, so the setting is not an unconditional free-RAM
guarantee. Elastic mode is implemented and tested but is not enabled in the
production launcher.

On Linux each stage buffer uses `mmap`/`munmap`, so reclaiming it actually releases
its pages instead of depending on whether the malloc allocator returns memory.
Windows keeps the upstream `VirtualAlloc`/`VirtualFree` allocation. The live-byte
getter takes the stage mutex, fixing its concurrent read of the watcher's counter.
Closing a source joins the watcher and clears the new session counters.

`STRATA_STAGE_KEEP_MIB` is the upstream alternative for configuring retention
directly. Unlike `--resident-lru-gib`, it does not subtract retention from the
resident profile budget; using it on top of an existing budget needs extra RAM.

`STRATA_TIER_TRACE=/path/to/file` writes each expert served outside GPU caches.
Letters R/F identify decode RAM/file reads, Q/P identify prompt RAM/file reads,
and r/f identify other fills. Trace output is optional. The serve log reports
decode and prompt LRU hits, and memory reclaimed by an elastic pool. Counters for
file bytes are more useful than file-blob lookups when assessing avoided I/O.

## Validation

Measured/tested on October 7, 2026: Linux, Ryzen 5700X3D, RTX 5070 12 GiB, 32 GiB
RAM, CUDA 13.4, IQ3_XXS read directly from its GGUF shards, NVFP4 KV and context
131072. The existing loaded service was stopped during rebuilding and GPU tests,
then restarted with the new binary and the same 22 GiB total expert budget.

- The fixed LRU test checks byte identity and destination guards, reuse without
  disk reads, eviction order, decode recency, prompt scans that do not admit or
  refresh entries, mixed cached/file batches and cleanup.
- The elastic test fills 832 canonical buffers (about 1.07 GiB), forces pressure
  through an unattainable availability floor, and checks reclamation to the
  1 GiB cap while prompt copies and statistics reads run concurrently. It checks
  protected pointers, at least 32 MiB returned in Linux VmRSS, reading an evicted
  expert, regrowth into freed slots and joining the watcher on close.
- Four CTest cases passed: file source, elastic LRU, VMM and KV/cache elasticity.
  The three GPU/RAM rotation variants also passed exact-byte checks.
- 27 server VRAM/runtime-info/vision tests passed. Eight native CLI checks rejected
  invalid sizes and LRU allocations without a larger total budget before loading.
- The real-model smoke test answered arithmetic correctly before and after an
  image, identified a red image, and completed two 256-token text generations.
  Decode measured 35.7 and 40.0 token/s for those two successive requests; the
  second benefited from warmed state. This is functional validation, not an A/B
  measurement of the PR's speedup. Logs confirmed 21,806 cumulative decode LRU
  hits after the text requests and restored all 3,354 GPU expert slots after
  the image. Linux native GGUF, on-demand vision and NVFP4 remained active.

The upstream Windows measurements are retained in
[their original report](../bench/results/2026-10-07-windows-ram-tier-lru/README.md).
They are not measurements of this machine. This change reduces repeated expert
reads; it does not establish or fix the internal cause of the NVMe latency spike
seen during the earlier live session.

[Local validation record](../bench/results/2026-10-07-ram-tier-lru/validation.json)
records the installed binary and checks. Private raw outputs are retained under
`/home/atef/others/strata-pr1324-integration/`.

## Local A/B

The same binary was compared with the LRU off (22 GiB by profile) and on
(18 GiB by profile + 4 GiB LRU), in fresh processes ordered A–B–B–A.
Each arm completed 16 measured text requests / 8,192 output tokens. Prompt
and conversation caches were disabled in both arms; thinking was disabled.
Uncached 8K and 32K prompts followed each process's decode workload.

The raw decode averages were 24.00 token/s off and 35.12 on. The last off
process suffered a roughly 69-second NVMe slowdown: median disk throughput
55 MiB/s, read await 39.8 ms and busy 97.5%. Including this episode gives
46.3% higher throughput with LRU, but does not establish that LRU prevents it.
All requests completed and all measurements remain in the primary result.

For a secondary balanced comparison, the affected request group was omitted
from all four processes, including its unaffected observations. The remaining
14 requests / 7,168 tokens per arm measured 30.44 versus 35.56 token/s
(+16.8%) and 31.4% fewer file bytes per token. Prefill measured 1198 versus
1107 token/s at 8K (-7.6%) and 1427 versus 1340 at 32K (-6.1%). Peak native
RSS was 26.70 GiB off and 26.12 GiB on, within the same expert-retention budget.
Generated text was not bit-identical, so this is a matched-task throughput
comparison, not a fixed-token replay or quality evaluation. Images were not
part of this benchmark.

The unchanged production launcher and service were restored with LRU enabled
and verified through the API. The [full A/B report](../bench/results/2026-10-07-ram-tier-lru-ab/README.md)
includes all four runs, the storage episode, per-topic results, output parity,
memory measurements and the sampling limits.
