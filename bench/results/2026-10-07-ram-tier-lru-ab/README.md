# RAM LRU A/B

Measured on October 7, 2026 on Linux, AMD Ryzen 7 5700X3D 8-Core Processor, RTX 5070 12 GiB,
32 GiB RAM and Predator SSD GM7 2TB firmware SN25043.
The model is the native GGUF Qwen3.8-Flash-Next IQ3_XXS, with NVFP4 KV,
KV/cache elasticity, MTP and the production control vector enabled.

## Method

Both arms use the same installed binary, SHA256
`bf3302e66e4f33e36d6769bafca62a35eacd0c1e98b471601100635651a25cee`. This compares the runtime allocation policy of
[PR #1324](https://github.com/Niko1221/Strata/pull/1324), not two source revisions.
The LRU and its Linux allocation changes exist in both arms' binary.

- **A:** `--resident-budget-gib 22`, no `--resident-lru-gib`:
  22 GiB by expert profile, previous stage retention policy.
- **B:** `--resident-budget-gib 22 --resident-lru-gib 4`:
  18 GiB by profile plus a fixed 4 GiB decode LRU.
- Fresh processes in order A–B–B–A. Same expert profile and native arguments,
  apart from the LRU flag. Stage retention overrides, elasticity environment
  variables and per-expert tracing are unset. All four processes use unbuffered
  file-tier reads. No system page-cache drop or SSD configuration change.
- Each process first generates 256 warmup tokens. These are reported in raw
  results but excluded from the decode aggregates.
- Four identical Italian prompts: gardening, Linux inference systems, physics,
  European history. Each requests 512 output tokens, repeated in the same order
  twice per process. Each arm completes 16 measured requests / 8,192 tokens.
- Prompt and conversation caches are disabled in both arms. Identical uncached
  8,192- and 32,768-token archive prompts follow each process's decode workload.
  All eight long-prompt answers correctly recall the same number (731).
- Same greedy sampling (`temperature=0`, `seed=7`), with thinking disabled.
  The harness calls the production `StrataEngine` directly. HTTP, vision,
  conversation snapshot overhead and prompt-cache reuse are outside this test.
- One-second sampling of native RSS, available RAM, process and NVMe I/O,
  CPU and GPU. Decode bytes use the native DONE `file_mb` counter, which counts
  actual reads and begins after prefill. Process read bytes include prefill.
  File-blob lookup counts alone would also count LRU-served lookups.
- Throughput is total output tokens divided by total native decode time;
  prefill throughput is total input tokens divided by total native prompt time.
  All measured runs are retained.

## Results

| Metric | A: profile 22 GiB | B: profile 18 + LRU 4 GiB | Change |
|---|---:|---:|---:|
| Decode, token/s | 24.00 | 35.12 | +46.3% |
| Actual file-tier MB / decoded token | 42.61 | 29.25 | -31.3% |
| Total time for 16 short requests, seconds | 355.01 | 247.84 | -30.2% |
| Mean first-token latency, seconds | 0.922 | 0.978 | +6.1% |
| Peak native RSS, GiB (maximum of two runs) | 26.699 | 26.118 | -0.582 GiB |

Per process, including prefill in RSS peaks:

| Order | Arm | Decode token/s | File MB/token | Peak native RSS GiB |
|---|---|---:|---:|---:|
| 1 | A | 29.67 | 42.98 | 26.699 |
| 2 | B | 35.57 | 28.95 | 26.111 |
| 3 | B | 34.68 | 29.55 | 26.118 |
| 4 | A | 20.15 | 42.23 | 26.679 |

## Storage episode and matched sensitivity

The last A process encountered an NVMe latency episode during `science-2`:
512 tokens took 86.56 seconds (5.91 token/s), rather than approximately
19 seconds. One-second samples identify 68.74 seconds with
read await above 10 ms and device busy above 80%. Median disk throughput was
54.72 MiB/s, await
39.81 ms and busy
97.54%. Median sampled output was
0.96 token/s. This happened around
17:07–17:08 Europe/Rome and resembles the earlier live NVMe slowdown.
No B decode samples met the same slow-disk criterion.

The raw +46.3% speed difference includes this
episode; it is not a clean estimate of the LRU's normal speedup. A single
episode in A and none in B cannot establish that the LRU prevents the SSD event.
The engine continued and completed every request without a crash.

For a **secondary, balanced sensitivity calculation**, omit the `science-2`
request group from **all four processes**, including its unaffected A and B
observations. This leaves the same 14 requests / 7,168 tokens in each arm.
No record is removed from the primary aggregates or raw data.

| Matched groups without the affected request group | A | B | Change |
|---|---:|---:|---:|
| Decode token/s | 30.44 | 35.56 | +16.8% |
| File-tier MB/token | 41.38 | 28.39 | -31.4% |

The unaffected full A process alone measured 29.67
token/s; the two B processes measured 35.57 and
34.68. This supports a generation benefit, but the
sample is too small to measure the probability of SSD slowdowns.

Per topic (both passes and both processes):

| Topic | A token/s | B token/s | Change |
|---|---:|---:|---:|
| garden | 30.88 | 36.66 | +18.7% |
| systems | 33.50 | 40.10 | +19.7% |
| science | 14.34 | 32.24 | +124.9% |
| history | 28.84 | 32.62 | +13.1% |

Uncached prefill:

| Tokens | A token/s | B token/s | Change | A individual runs | B individual runs |
|---|---:|---:|---:|---|---|
| 8192 | 1198.4 | 1107.0 | -7.6% | 1187.6, 1209.4 | 1142.7, 1073.4 |
| 32768 | 1427.2 | 1340.4 | -6.1% | 1423.6, 1430.8 | 1338.9, 1341.9 |

Each process starts with 3,354 GPU expert slots, retains 3,354 after 8K,
and has 3,256 after KV grows for 32K. The slot changes match in both arms.

## Interpretation and limits

The fixed 4 GiB LRU is useful for these text workloads: it avoids repeated
expert reads and improves generation at the same expert-retention budget.
The matched sensitivity gives +16.8% generation speed
and -31.4% file bytes per token;
the full raw average is also affected by the A-only NVMe latency episode.
It trades some prefill speed for that gain because fewer experts are selected
statically by the profile. It does not require 4 GiB on top of the 22 GiB budget.
Native RSS also includes weights, mappings and transient allocations; this
test does not include the production conversation snapshots.

Outputs are not bit-identical: only 2 of 10 measured
request groups have identical output across all four processes. Decode lengths
are identical (512 each), and the recall answers are correct. The upstream
[runtime documentation](../../../docs/DETAILS.md) already describes how expert
residency and rounding can change future tokens. This is a matched-task
end-to-end throughput comparison, not a fixed-token replay or a quality study.

There are two process repetitions per arm and synthetic text prompts. These
measurements do not establish the gain for images, 131K context, every prompt,
or a guarantee against the previously observed 66-second NVMe slowdown.
The longest measured inter-token gaps are 4.285 seconds in A
and 0.176 seconds in B. Startup/warmup and prefill time are excluded
from those gap figures and from the decode speed aggregates.

The production service was restored using the unchanged minimal launcher,
with 22 GiB total / 4 GiB LRU. The benchmark itself did not publish a release.

[Summary](summary.json) includes individual prefill runs, per-topic results,
memory and disk samples, and output-parity counts.
[Environment](environment.json) records the build and hardware.
Raw results, synthetic outputs, prompts, timings, logs and the benchmark
harness are retained privately in `/home/atef/others/strata-pr1324-ab`.
