# NVFP4 exact kernel optimizations and main integration

2026-10-06. The feature is opt-in (`--kv nvfp4 --kv-resident 0`); existing
INT8 Bash profiles are not changed by this integration. Main K/V use the fixed
RHT256 / row148 codec; the MTP drafter remains INT8. No native FP4 MMA is claimed.

## Exact computation changes

- Four E2M1 codes are unpacked into exact signed integer bytes with byte
  permutations and byte subtraction; a scalar host/HIP fallback is retained.
- Sixteen shared value-scale maxima are calculated once rather than repeatedly
  by each consumer. Float division/multiply and MMA accumulation order remain.
- NVFP4 query hi/lo half fragments are loaded into registers once before the
  selected-cell loop. Other storage modes retain their existing query loads.

The checked SM120 kernel uses 252 registers, no local spill/stack storage;
this resource use is not a performance guarantee for other architectures.

## Measurements before integration

RTX 5070 12GB, Ryzen 7 5700X3D, 32GiB RAM, CUDA 12.8. Q3 ISTA plus the optional
Abl direction, 128k allocated context, chunk8192. Actual Q3 inputs reached
32261 tokens, not full128k. Numerical attribution used greedy sampling,
static mapping of2378 GPU experts and no resident RAM expert cache.

An alternating CUDA-event microbenchmark at batch8192 measured original
NVFP4 attention61.97ms, optimized51.82ms (16.38% less kernel latency), versus
INT826.05ms. In one whole-Q3 profiled ordering, prompt attention fell from
1412.28ms to1180.59ms (16.41%). Complete prefill was not faster: INT8260.7,
original NVFP4246.6, optimized NVFP4241.4tok/s. These are single instrumented
observations, not reliable end-to-end gain/regression estimates. Mapped weight
reads, staging and issuer/consumer waits dominate. Phase label `dequant`
includes compute-stream waiting, not just dequantization-kernel execution.
Nsight used CUDA12.8 libraries and warned about unsupported driver13.4;
no dropped events were reported, but trace inactivity is not certified GPU idle.

## Numerical and generation checks before integration

The isolated optimization matched frozen NVFP4 prompt outputs bit-for-bit
through batch8192. The CUDA parity harness checked22 geometries and7097117
conditions, with invalid sparse IDs, selected widths, page sizes and canaries.
Unpack checked662144 signed codes; memcheck/racecheck/initcheck/synccheck were
clean. Nine no-thinking and one high-thinking complete generated token sequences
matched frozen NVFP4, with bit-identical first248320-element logit vectors.
This does not prove every intermediate autoregressive logit is bit-identical.

Relative to INT8, NVFP4 is lossy:32 sampled matched-reference-context predictions
had30 equal top1 tokens, mean KL0.00187255nats and maximum0.01610157nats.
They are sampled re-prefills, not a perplexity or accuracy benchmark. One INT8
re-prefill also differed from its own free-running reference. Four high-thinking
prompt pairs gave identical correct final answers; three thinking trajectories
differed. Both formats had inaccurate open technical prose in one fixture.
No general quality-equivalence or model-accuracy claim is made.

## Integration validation on local main

The clean-main CUDA12.8/SM120 Release binary is installed at `build/strata`;
the reproducible build directory is `build-nvfp4-main`. It was compiled from
an NVFP4-only main source snapshot, not from the unrelated dirty worktree.
Pre-existing local staging/watchdog and server-default edits remain separate
and uncommitted; the new native binary does not promote the staging candidate.
The old installed binary is privately backed up. No push, systemd/autostart,
production server launch or profile quantization switch was performed.

- 8 complete Q3 Abl CLI generations (INT8/NVFP4; arithmetic, code,32k retrieval,
  high-effort code) matched their corresponding frozen output token IDs and
  first248320-float logit vectors bit-for-bit. Actual maximum input32225.
- 16 Q3 Abl HTTP/SSE requests passed known answers and native/API usage counters.
  Exact repeat, append and A/B/A RAM return reused over99% of the prompt;
  the final return reused17259/17260 tokens in both formats.
  Both test servers exited0 without forced kill; test ports were left stopped.
- These API trials used128k allocation,8k reads/checkpoints, fixed2378 GPU
  expert slots, resident16GiB and adaptation96. They do not validate the
  owner's unchanged resident27GiB profile or a full128k-token input.
- All three NVFP4 device gates pass, including22917 native-conversion pairs,
  7097073 GPU checks and662144 unpack code checks. All12 combinations of
  device executable and memcheck/racecheck/initcheck/synccheck are clean.
  A separate frozen-reference parity gate passes7097117 checks.
- The canonical-main kernel microbenchmark (8 alternating rounds per arm)
  measured original NVFP461.9416ms, optimized51.8973ms at batch8192:
  16.22% less attention-kernel latency. INT8 was26.0195ms. This is not a
  16% end-to-end prefill gain.
- Host-only suite17/17 passes; codec/snapshot ASan+UBSan2/2 pass. The server
  suite246 tests passes with4 skips. Six source-contract and3 CLI tests pass.
- Full native CTest:64 pass,2 skip,2 environmental/legacy failures remain
  visible. `ple_parity` expects the removed Q2 GGUF/oracle fixtures;
  `platform_memory_test` requires256MiB mlock, above the current8MiB hard limit.
  The latter is reproduced on baseline; affected sources are byte-unchanged.
  No OS limits were changed and these failures were not relabeled as passes.
- Setup golden fixtures retain exactly the same46 failure identifiers on
  baseline and candidate. NVFP4 setup5/5 and logit-metric17/17 tests pass.

Existing INT8 Bash launchers and their authentication/network/resource options
are unchanged. To opt in, explicitly use `--kv nvfp4 --kv-resident 0` in native
options. Codec/layout/seed and cache compatibility boundaries are unchanged
from the supplied NVFP4 candidate; a rebuilt binary can make old snapshots
foreign and the first request cold. Existing production caches are not deleted.

Quality/divergence and kernel gains are bounded observations, not a general
quality-equivalence, global bitwise-logit, HIP-support or total-prefill recovery
claim. Whole-prefill expert staging/paging optimization remains separate work.
