# Owner integration of Strata 0.1.40.1

The fork incorporates official `v0.1.40.1`, commit
`82f46a8c8f475f001ad76d92f58f4a4f8ffb0253`, while retaining the owner changes
at `f04487093d0fc5c510cacbc03191d5a89bab96e7`. The engine reports `0.1.40`,
as the official hotfix changes the Python server rather than the engine version.
The fork release is `v0.1.40.1-nvfp4.1`.

Upstream rewrote its Git history. The merge uses the verified common 0.1.38
tree `b430923af1caa6c1d497c33364e7a6f8e07ef834`: owner `99f3dbd` and
official `09817da` have exactly that tree. Three-way merging against that tree
preserves both histories and both parents without replacing the owner source.

## Preserved behavior and integration

- NVFP4 main KV remains available through `--kv nvfp4 --kv-resident 0`.
  The codec, RHT256, distinct snapshot format, bounds checks and exact arithmetic
  settings remain. MTP uses INT8; the default main format remains INT8.
- Balanced/root/fork checkpoints, buffer recycling, phase measurements,
  RAM admission and snapshot validation remain. Upstream tail checkpoints,
  message boundaries, nullable draft state and stage-specific snapshots remain.
  The owner RAM switches remain `--conversation-ram-cache-*`; legacy disk
  switches keep their existing meanings and optional OpenSSL dependency.
- Direct server CLI followed by native arguments after `--`, epoch-based STOP,
  disconnect cancellation, FIFO status ownership, cache counters and Anthropic
  effort mapping remain. The official hotfix for quoted tool calls and waiting
  requests during engine restart is included. DONE accepts both upstream tier
  counters and named owner coverage/phase fields.
- Upstream AVX2/VNNI dispatch remains; the owner's IQ4_NL cached scale work now
  lives in the shared rows implementation. Upstream batching, elastic VRAM,
  KV growth, streaming hybrid KV and multi-GPU source are retained.
- NVFP4 uses its per-token append when upstream's batched append cannot encode
  this format. Elastic KV allocates the two NVFP4 arrays with the correct row
  size, including clearing newly mapped memory and limiting reset to mapped cells.
- MTP capture errors report the failing CUDA operation and its actual status.

## Watchdog correction

The first long-input trial reached 67,121 tokens, then the watchdog aborted a
verify window after 60 seconds without a token/window heartbeat. Its diagnostic
reported 29 completed layers, then advancement to layer 48 two seconds later.
The host was still completing work; this instance does not establish the cause
of every earlier stall.

Completed expert layers now also advance the heartbeat. Merely changing the
stage does not. Waiting indefinitely on an unchanged GPU flag still reaches
the watchdog limit. The timeout stays at 60 seconds and is not disabled.

## Validation on 2026-10-07

Measured on Linux x86_64, Ryzen 7 5700X3D (AVX2), 32 GiB RAM and RTX 5070
12 GiB (SM120), built with CUDA 13.4 in Release mode, native experts enabled.

- The full Python server suite passed: 498 tests, with only three Windows-only
  tests skipped. This includes the new combined upstream/owner DONE regression,
  the real tokenizer tests and JSON Schema checks.
- Tools: 509 tests passed, 4 skips; the entire suite also passed under `python -O`.
  The separate request capture test passed. The CI host build passed all 26
  registered tests. The standalone host suite passed all 16 under ASan/UBSan.
- All 120 native tests were attempted: 117 returned success and three failed.
  One failure, the memory-lock test, passed in the owner's privileged retest.
  The remaining two require unavailable PLE fixtures and AVX-512 hardware.
  The success count includes `expert_parity`, `pool_test` and `pool_stress`,
  which explicitly skip their unsupported AVX-512 paths; these paths were not tested.
  The available CUDA and AVX2 parity, cache, snapshot and cancellation gates passed.
- NVFP4 GPU tests exercised 7,097,073 checks across 22 fixtures; unpacking
  exercised 662,144 signed-code checks. Conversion, attention, unpacking and
  elastic-pool tests were checked with memcheck, racecheck, initcheck and synccheck.
- Batched native PLE matched sequential postops at sizes 1, 2, 8, 9, 10, 64,
  65, 77, 256 and 1024 with maximum absolute error zero. Size 1024 also passed
  CUDA memcheck. The separate mmap expert-source regression passed.
- Real IQ3_XXS + MTP + NVFP4 requests covered short/repeat/changed-tail cases,
  a 67,128-token input, and its 67,127-token fork with 65,536 cached tokens.
  The long request completed in 52.796 seconds; the fork in 3.396 seconds.
- A RAM A/B/A trial restored 9,919 of 9,926 prompt tokens (99.93%) and returned
  the expected answer. Streaming disconnect, the following request, Anthropic
  Messages and OpenAI Responses also returned the expected results.

The long-context trial used 131,072 allocated context, 640 MiB VRAM reserve,
8192-token prefill/checkpoints and 22 GiB resident experts. The RAM admission
trial used 16 GiB resident experts to leave space for parked conversations.
The deployed launcher keeps the owner's 22 GiB setting and 640 MiB reserve.
The engine reported 223 MiB free VRAM after loading; this remains a tight profile.
Physical admission can reject parking when the 4096 MiB RAM floor would be crossed.

## Limits of the measurements

The original PLE oracle test still needs its unpublished `ple_in.bin`,
`ple_out.bin` and canonical pack/model fixtures. The separate native PLE tests
are additional evidence, not a replacement for that independent oracle.
`expert_multi_test` specifically needs AVX-512/VNNI, absent on this CPU.
Neither unavailable test is relabeled as a pass. The 256 MiB mlock test passed
when the owner ran it with a per-process 512 MiB limit under sudo.

No full 128k input, HIP, SYCL, multi-GPU runtime, universal binary portability,
full-model logit parity or general throughput improvement is claimed.

## Rebuild

Use the repository's pinned llama.cpp source, a CUDA 13 toolkit and OpenSSL 3
development headers for the optional owner disk cache:

```sh
cmake -S . -B build-custom -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_NATIVE_EXPERTS=ON \
  -DSTRATA_GGML_DIR="$PWD/third_party/llama.cpp" \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DSTRATA_BUILD_TESTS=ON \
  -DSTRATA_BUILD_CONVERSATION_TESTS=ON \
  -DSTRATA_ENABLE_CONVERSATION_DISK=ON -DSTRATA_TEST_NVFP4=ON
cmake --build build-custom -j8
ctest --test-dir build-custom --output-on-failure -j1
bash tools/run_nvfp4_gpu_checks.sh build-custom
```

The released executable targets SM120 and AVX2/FMA/F16C. It requires CUDA 13
cuBLAS/cuBLASLt/cudart, OpenSSL 3, glibc >=2.38 and GLIBCXX_3.4.32.
Runtime libraries, model weights and private launchers are not bundled.

The machine-readable [validation report](../bench/results/2026-10-07-nvfp4-0.1.40.1/validation.json)
records the test counts, hardware, unavailable gates and released engine checksum.
