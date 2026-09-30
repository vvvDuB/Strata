# Fork customizations and upstream integration

This fork retains the local inference/serving changes made on Strata 0.1.20
(`c1e903310f211e6630780c3bd2038778c071c68d`) and integrates official 0.1.28
(`bbaaabb4643bb7873d4cef9d48b5dcf96e6cbff4`). The pre-merge custom snapshot is
`cca1368273f7798350cba3c1a4fc84fb0012363b`.

## Retained features

- Balanced checkpoint retention, sparse coverage, bounded system/root pins,
  latest leaf, one learned exact text-fork boundary, and checkpoint buffer reuse.
  `--prompt-cache-policy lru` remains available for legacy eviction behavior.
- Exact checkpoint-boundary planning: reads stop at interval, root, turn and
  learned fork boundaries. Cache states after a changed token are never reused.
- Optional system-prefix persistence (`--prefix-cache-file`) and a separate
  multi-conversation disk store (`--conversation-cache-dir`,
  `--conversation-cache-mib`, `--conversation-cache-slots`). Neither is mandatory.
  The Q3 deployment uses the conversation store, not dedicated Claude-prefix
  persistence. Disk snapshots contain prompt tokens and model state: protect
  their directory and never commit them.
- HTTP disconnect and queued-request cancellation, per-layer prefill STOP,
  epoch-based STOP tickets, and FIFO-owned request finalization. A queued
  cancelled request must not clear another request's active status.
- OpenAI/Anthropic cached-token counters and native common-prefix/replay-gap
  diagnostics. Official request timings and status/privacy fixes are retained.
- Direct Python-wrapper CLI (`--exe`, `--cwd`, `--log`, `--alias`, `--lib-dir`,
  `--tokenizer`) followed by native arguments after `--`. No external model JSON
  is required; legacy `--config` remains an alternative, not an additional source.
  Single numeric GPU selection and official comma-separated GPU lists both work.
- Built-in Anthropic effort mapping: low/medium unchanged, high/xhigh/max mapped
  to xhigh, explicit disabled thinking wins. No additional effort flag.
- CUDA 12.8/SM 120 PLE batched-convolution unrolling fix and its GPU regression.
  Official native mmap expert layout validation now includes the previous local
  offset/size/unmap fix; **both** local and official expert-source tests remain.

## Integration decisions and limitations

Official CUDA/HIP kernels, threaded expert-copy issuance, fused prefill and
multi-GPU implementation are retained. Cancellation signals and joins the copy
issuer before finishing its stager and draining GPU streams.

Exact-boundary reads finish/join all stages before saving a checkpoint. Therefore
multi-GPU checkpoints use synchronously captured parts at that boundary, rather
than assembling parts from stages already at different pipeline positions.
Recycled checkpoint buffers clear obsolete stage parts before reuse. This can
change the pipeline's throughput; multi-GPU performance is **not validated**.

The optional v1 disk-state codec still requires Linux, **single-GPU resident
INT8 KV**, MTP, prompt cache/root, and no vision/control vectors. Streaming KV
(`--kv-resident` nonzero) and multi-GPU disk persistence are explicitly rejected;
ordinary upstream operation without these optional disk flags remains available.
No multi-GPU/HIP runtime test or full 128k-input test is claimed.

The pinned GGML CUDA MMQ sources reference `cuGetErrorString`. Their CUDA-only
CMake target now links `CUDA::cuda_driver` as well as cudart; the missing driver
link was exposed by a clean Linux CUDA build.

## Reproducible validation

Build independently of an existing installation (CUDA 12.8, GNU C++ 14.2,
RTX 5070 / SM 120 were used):

```sh
cmake -S . -B build-custom -DSTRATA_ENABLE_CUDA=ON \
  -DSTRATA_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build build-custom --target strata native_ple_batch_test \
  file_expert_source_test local_file_expert_source_test -j4
python3 -m unittest discover -s serve -p 'test_*.py'
python3 -m unittest discover -s tests -p 'test_*.py'
cmake -S tests -B build-cache-tests
cmake --build build-cache-tests -j4
ctest --test-dir build-cache-tests --output-on-failure
build-custom/file_expert_source_test
build-custom/local_file_expert_source_test /tmp/strata-mmap-new-directory
```

The last command needs a previously nonexistent directory. The standalone cache
suite does not require the unpublished full upstream oracle/test sources.

Validation on 2026-09-30: 97 serving Python tests (3 opt-in tests skipped), one
request-capture test, five standalone C++ cache suites, both expert-source suites,
and a successful native CUDA build. The five cache suites also run under
ASan/UBSan. PLE batches 1, 2, 8, 9, 10, 64, 65, 77, 256 and 1024 have zero
compute-sanitizer errors and zero absolute difference from sequential postops.

Live Q3 integration tests use the retained IQ3_XXS model, 131072 allocated
context, resident INT8 KV, mmap experts, MTP spec 4, prefill 4096, 12 balanced
checkpoints every 4096 tokens with root pinning, and a separate four-slot / 8192
MiB disk namespace. They launch through a private Bash profile on localhost,
not a systemd service. Model weights, authentication keys, local profiles, cache
files and raw logs are not included in this repository.

Old disk caches are build/config-identity checked; a new engine build can make
the first request cold. Cache ratio depends on an **exact token prefix**, not
semantic similarity. Editing before the first interval may initially reuse
only the root; the learned fork boundary reduces subsequent replay. Expert
weight-cache hit percentage is not the conversation/prompt-cache hit percentage.

### Live Q3 results (single run)

| Case | Input tokens | Reused | First output (s) |
| --- | ---: | ---: | ---: |
| Cold input | 9315 | 0 | 54.897 |
| Exact repeat | 9315 | 9308 | 0.305 |
| Appended turn | 9338 | 9319 | 0.577 |
| First history edit before interval 4096 | 9338 | 21 | 50.456 |
| Second edit at the learned fork | 9338 | 3077 | 33.346 |
| Different system root | 9338 | 0 | 49.243 |
| Return via disk snapshot | 9338 | 9331 | 4.294 |

Anthropic A/B/A/B returned the correct synthetic project codes and reused
414 / 390 tokens on returning to each branch. A fresh long prefill was
interrupted: native per-layer cancellation was observed 0.201 seconds after
socket closure and the correct replacement answer completed after 2.009 seconds.
This measures cancellation, **not completion of that long input**. It is not a
throughput comparison against the old engine and not a Pi/Claude capture test.
Counts-only evidence is in
`bench/results/2026-09-30-fork-integration/summary.json`; raw HTTP/log evidence
remains private outside the repository.

## Shared-cache integration follow-up

See [CACHE_INTEGRATION.md](CACHE_INTEGRATION.md) for the compatible #189/#190
integration and opt-in #203 staging adaptation. The existing disk CLI retains
its meaning; RAM parking uses `--conversation-ram-cache-*`. Checkpoint payload
identity is now v2 (five buffers, including indexer accumulator and position),
so use a fresh cache namespace for validation. Earlier live-model results above
belong to the previous build and are not validation of this integration.
