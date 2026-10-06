# Resident RHT256 + NVFP4 KV

Status: experimental and opt-in. CUDA 12.8 / RTX 5070 SM120 execution, bounded
Q3 generation, cache behavior and isolated kernel optimizations have been tested.
Canonical-main integration validation is complete on the tested host. HIP, other hardware and
full 128k-input validation are not claimed. See [NVFP4_OPTIMIZATION.md](NVFP4_OPTIMIZATION.md).
Date: 2026-10-06.

## Scope and numerical contract

`--kv nvfp4` selects K-NVFP4 and V-NVFP4 for the main QSA layers. The implementation
combines a randomized orthogonal Hadamard transform with the NVFP4 numerical
format. It is TurboQuant-inspired, not the full TurboQuant algorithm: there is
no Lloyd-Max codebook, QJL correction, stochastic rounding or retraining.

The code targets the input engine geometry: head dimension 256 and 12 query
heads per KV head. The supplied Qwen configuration has 2 KV heads and 12 QSA
layers. Main KV is resident (`--kv-resident 0`); streaming and ring storage are
rejected. The MTP drafter keeps INT8. Indexer storage/math and Gated DeltaNet
storage/math are unchanged. Model hidden states can still differ after KV
quantization, so this does not promise identical downstream indexer selections.

The new path does not alter the model weights, the sparse selection budget,
expert residency policy or default quantization. `STRATA_KV_ROT` belongs to the
legacy INT8 path; NVFP4 always uses its own fixed RHT, once only.

Let H be the normalized 256-dimensional Hadamard matrix and D a fixed sign
diagonal. This implementation uses R = H D:

- K: normalize, apply RoPE, then R and NVFP4 encode;
- V: R and NVFP4 encode;
- Q: normalize and apply RoPE, then R, without FP4 quantization;
- attention result: apply R transpose = D H, then the original output gate and projection.

Before quantization, `(R q)^T (R k) = q^T k`. The inverse is **D H**, not another
forward H D. Quantized keys affect attention scores; quantized values affect the
weighted sum. Neither logit identity nor quality neutrality is guaranteed.

The fixed hash/seed is serialized behavior: `0xa341316c`, with multipliers
`0x7feb352d` and `0x846ca68b`. Changing these, normalization, nibble order, scale
rules or row layout requires a new cache format/identity.

## Wire layout and scaling

Each `(token, KV head, K-or-V)` is an independent dynamically scaled NVFP4 tensor.
It is not one global scale shared over the growing conversation. This permits
append without requantizing historical rows.

```text
Row, alignas(4), sizeof = 148 bytes
  [0,128)    packed E2M1 codes, even component in low nibble
  [128,144)  16 nonnegative E4M3 scales, one per 16 components
  [144,148)  FP32 tensor scale
```

For a finite, nonzero rotated row:

```text
global_scale = max(row_amax / (448 * 6), FLT_MIN)
block_scale  = round_E4M3((block_amax / global_scale) / 6)
code[d]      = round_E2M1((rotated[d] / global_scale) / decoded(block_scale))
x_hat[d]     = decoded(code[d]) * decoded(block_scale) * global_scale
```

The encoder uses the rounded block scale when computing codes. Both conversions
use round-to-nearest-even. Signed FP4 zero is preserved; zero-scale blocks
produce zero codes. A zero row has zero tensor scale. Invalid input rows carry
NaN in the tensor scale rather than silently producing a plausible valid row.
The portable scalar conversion has SATFINITE behavior, tested against an
independent nearest-level oracle. Extremely small scales are handled explicitly;
this is not a claim of bitwise behavior across all device floating-point modes.

Native FP4 conversion is gated on CUDA >= 12.8 and device compilation architecture
>= 1000. It calls `__nv_cvt_float2_to_fp4x2` from `cuda_fp4.h` with E2M1 and
`cudaRoundNearest`. Other builds use the scalar conversion. The GPU conversion
test includes nibble-order, signed midpoint, saturation and dense pair checks.
Do not infer that this branch works on your GPU merely because the host codec passes.

The row is a Strata-specific paged layout using NVFP4 numeric encodings. It is
not a drop-in padded scale-tensor layout for Transformer Engine, TensorRT-LLM
or FlashInfer block-scaled GEMM. A future native FP4 MMA path needs an explicit
layout/operand design and independent tests.

## Kernel implementation

### Append and rotation

`src/kernels/cuda/kv_nvfp4.cu` uses four complete warps per CTA, one warp per row.
The 256-component transform is a register/shuffle butterfly, with no shared
scratch or CTA-wide barriers. Tail exits are warp-uniform. Each K/V row has a
single writer; this avoids the duplicate-alias writes used in some older hybrid
paths. Inputs remain unmodified. Exact in-place and disjoint out-of-place query
rotation are supported; partially overlapping source/destination buffers are not.

The step append reads the token position from device `step`, so a captured CUDA
Graph does not freeze the first position. A separate batched append accepts host
position/count. Invalid append positions or physical pages trap rather than
writing outside the pool. Page ownership/table validity remain caller invariants.
Resident tables are unique mappings; aliased physical pages for concurrent
logical rows are unsupported.

Rows have only four-byte alignment. Decode and prompt loads deliberately use
aligned 32-bit reads, never assume each 148-byte row is 8/16-byte aligned. Shared
prompt rows have separately aligned vectorized writes. Compile options keep the
append/rotation translation unit out of flush-to-zero and approximate division.

### Decode

`qsa_decode_attn.cu`, mode 5, reads selected rows directly from compressed pools.
K is decoded to FP32 registers, scores/softmax/value accumulation follow the
existing FP32 decode structure, and the result remains in the rotated basis
until the caller applies the inverse. Negative/out-of-range selected cells and
invalid physical pages are masked in this direct attention kernel. The device
selection width is clamped to the allocated capacity.

### Prefill and speculative verification

`qsa_prompt_attn.cu`, mode 5, gathers only the current selected chunk into shared
memory. Each E2M1 code becomes the exact signed integer `2 * E2M1`, in
`{0, +/-1, +/-2, +/-3, +/-4, +/-6, +/-8, +/-12}`. These integers are exactly
representable by INT8 and FP16. One FP32 scale per 16 values carries the half
factor and both NVFP4 scales. The prompt kernel applies scales to FP32 partial
products, retaining the existing hi/lo FP16 split for queries and probabilities.
It does not first round all scaled K/V to FP16 and does not allocate a
full-context dequantization buffer. Sparse-invalid cells are masked before softmax.

This is **FP16 MMA with FP32 scaling/accumulation**, not native FP4 QK/PV MMA.
The extra scale groups and transforms have a measured kernel cost on SM120.
The exact unpack, shared scale reduction and query-fragment register cache reduce
this cost without changing the checked NVFP4 outputs. NVFP4 storage density
alone does not imply an end-to-end speedup; see the optimization report.

A generic selected-row FP16 gather is available when the existing engine selects
its fallback attention path. That fallback does materialize the selected rows
in FP16, and retains the original attention arithmetic. Invalid gather IDs emit
zeros without out-of-bounds reads; as with the original gather-based path,
mathematically valid attention requires valid selected IDs from the indexer.
The optimized direct kernels provide their own invalid-ID masks.

The real-device test gates both decode and SM80+ prompt execution. HIP and old
GPU fallbacks are implemented structurally but not validated in this environment.
The shipping target of this experiment is the user's single RTX 5070.

## Engine integration and persistence

The new state flag, pointers and byte accounting are threaded through main
allocation/zeroing, decode, batched prefill, speculative verification, snapshot
capture/restore, prefix cache records, disk identity, CLI, launcher and web label.
MTP initialization temporarily selects INT8 with RAII flag restoration, including
failure exits. Existing INT8, FP16, Q4_0 and hybrid paths remain available.

Snapshot format is 5. Format 4 remains reserved for the earlier experimental
TQ4-V artifact and is rejected here; it is not interpreted as NVFP4. The identity
marker is `nvfp4-rht256-v1-seed-a341316c-row148`. Main NVFP4 pools serialize as two
arrays; INT8 MTP keeps its original four arrays. Cross-format restore is rejected
before target writes, and incremental capture checks its previous format before
changing metadata. Prefix layout checks include row capacity and distinct K/V pools.

Keep old binaries and caches for rollback. Use separate cache filenames/directories
for A/B tests. Do not rename/convert a saved INT8 or old TQ4 file into NVFP4.
Existing disk-cache restrictions (single GPU, MTP, prompt cache/root, no vision or
control vectors) are retained. Live GPU A/B/A restore and speculation acceptance
still need end-to-end testing.

`setup.py --kv nvfp4` requires `--build`, preserves an explicit NVFP4 choice even
below 8K context, and forces resident main KV. The MCP install schema is unchanged
because it has no local-build argument. Default profile generation remains
byte-identical in the 25 compared fixtures.

## Memory accounting

Per main layer and context token: `2 KV heads * 2 pools * 148 = 592 bytes`,
compared with 1056 for the input engine's grouped INT8 codec. Do not substitute
a generic GGUF Q8_0 estimate for this engine-specific layout.

| Context tokens | INT8 main K/V GiB | NVFP4 main K/V GiB | Saved MiB |
|---:|---:|---:|---:|
| 65,536 | 0.77344 | 0.43359 | 348.00 |
| 124,000 | 1.46341 | 0.82040 | 658.45 |
| 131,072 | 1.54688 | 0.86719 | 696.00 |
| 262,144 | 3.09375 | 1.73438 | 1,392.00 |

These are calculated payload sizes over 12 main QSA layers: a 43.94% reduction.
They exclude page rounding, allocator alignment, indexer, GDN, MTP, rope,
attention scratch, expert cache and weights. Per-vector tensor scaling costs
4 additional bytes/head; quoting 144 bytes/head would undercount this format.
The launcher includes INT8 MTP in its estimate: `12*592 + 1056 = 8160 B/token`.
Actual reserved VRAM, peak VRAM and expert hit-rate must be measured separately.

## Build and tests

### Host-only tests, no toolkit or weights

```bash
cmake -S tests -B build-nvfp4-host -DCMAKE_BUILD_TYPE=Release
cmake --build build-nvfp4-host --parallel
ctest --test-dir build-nvfp4-host --output-on-failure
python tools/test_nvfp4_compare_logits.py
python tools/test_setup_nvfp4.py
```

Snapshot host tests link actual snapshot/prefix source against an isolated memcpy
transport stub in `tests/host_cuda_stub`. This is not GPU emulation and does not
exercise device concurrency, driver errors or kernels. The stub directory is
never in production include paths. The Python comparator requires NumPy.

### CUDA kernel gates without model/GGML

Use a toolkit supporting SM120. Add your normal compiler/toolchain options as
needed. The following disables native expert integration only for this dedicated
kernel-test build; it is not the recommended engine configuration:

```bash
cmake -S . -B build-nvfp4-kernels \
  -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DSTRATA_NATIVE_EXPERTS=OFF \
  -DSTRATA_TEST_NVFP4=ON
cmake --build build-nvfp4-kernels \
  --target kv_nvfp4_gpu_test kv_nvfp4_native_conversion --parallel
bash tools/run_nvfp4_gpu_checks.sh build-nvfp4-kernels
```

For an engine build, retain your working Strata GGML/native options and build
`strata` plus the two tests. `-DSTRATA_TEST_NVFP4=ON` is off by default. No
precompiled engine is supplied by this patch.

The GPU gate includes native FP4 conversion; CPU-vs-GPU rotation and fused append;
unchanged input buffers; guards around every allocation; page sizes 1, 4, 64;
permuted pages and partial tails; batch/decode/step/prompt attention against a
double-precision CPU reference over stored quantized values; zero/negative/oversize
widths; invalid cells/pages; 4099-cell fixture and a 2051-selection boundary;
selected gather; inverse output rotation; and CUDA Graph append/decode replays
with changing device-side state. Reference parity is about implementation of the
chosen quantized attention, not its accuracy relative to the original model.

The Bash script runs both executables and all four Compute Sanitizer modes. It
fails on missing tools, nonzero test exit or skip. CTest records return 77 as
SKIPPED, so a green CTest run with skips is not sufficient evidence. Native
conversion not exercised on a Blackwell device is also a skip/failure of the gate.

On Windows or multi-config generators, run the two executables from the actual
output directory, usually `build-nvfp4/Release`, then for **each** executable:

```text
compute-sanitizer --tool memcheck  --error-exitcode 99 <executable>
compute-sanitizer --tool racecheck --error-exitcode 99 <executable>
compute-sanitizer --tool initcheck --error-exitcode 99 <executable>
compute-sanitizer --tool synccheck --error-exitcode 99 <executable>
```

Do not relax thresholds or disable the native conversion check to label a failed
build as validated. Save compiler diagnostics, device model, toolkit/driver
versions and complete logs when a gate fails.

## Model-level comparison and performance

Use the same binary, weights, token IDs, RoPE, seed and settings; change only
`--kv int8` versus `--kv nvfp4`, with `--kv-resident 0` on both. Keep experts and
cache budget fixed for the first A/B. Only after isolating codec cost should the
freed VRAM be reassigned to experts in a separate experiment. Use a fresh process
and distinct persistent caches per format. Verify 32K, 64K and 124K prompts,
long retrieval, coding/reasoning, long generation, prefix resume and speculative
accept/reject. Record prefill/decode throughput, peak VRAM, attention time,
expert transfers and acceptance rate; report repeated runs, not one sample.

For native packs, the original engine's ordinary `--dump-logits` path is not
reached. Use its existing `STRATA_DUMP_FIRST_LOGITS` diagnostic instead. For
example, in Bash, append all your normal model/pack options to each invocation:

```bash
STRATA_DUMP_FIRST_LOGITS=baseline-first.f32 ./build-nvfp4/strata generate \
  --tokens-file prompt.ids --kv int8 --kv-resident 0 --max-new 1 <normal-model-options>
STRATA_DUMP_FIRST_LOGITS=candidate-first.f32 ./build-nvfp4/strata generate \
  --tokens-file prompt.ids --kv nvfp4 --kv-resident 0 --max-new 1 <same-model-options>
python tools/nvfp4_compare_logits.py baseline-first.f32 candidate-first.f32 \
  --raw-first --tokens-file prompt.ids --output first-logits.json
```

`<normal-model-options>` is a placeholder, not a literal shell argument. Use the
same verifier/speculative and prefill options in both runs. This raw mode measures
**one prediction after the full prompt**: top-1 agreement, KL and max logit delta;
it is not perplexity. Repeat across many representative prompts. Raw bytes cannot
establish that a file really came from the specified weights or token IDs.

Only for packs using the per-token dump path: use identical `--tokens-file`,
`--prefill 0 --spec 0 --max-new 1`, and matching `--dump-logits` and
`--logits-stride` options, then:

```bash
python tools/nvfp4_compare_logits.py baseline.logits candidate.logits \
  --tokens-file prompt.ids --stride 1 --output conditioned-metrics.json
```

That format has two little-endian int32 dimensions followed by float32 logits.
The tool refuses incomplete/nonfinite dumps or incompatible dimensions. A stride
above 1 reports only sampled conditioned positions, not full-corpus perplexity.
There is no universal top-1/KL threshold that certifies quality neutrality.

## Sources and boundaries

NVFP4 E2M1 / block E4M3 / tensor FP32 representation and scale constants:
NVIDIA Transformer Engine 2.15 documentation,
https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.15/user-guide/features/low_precision_training/nvfp4/nvfp4.html

Native pair conversion signature, rounding argument and saturation:
CUDA Math API 12.8.1,
https://docs.nvidia.com/cuda/archive/12.8.1/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP4__MISC.html

The 256-dimensional RHT, per-row tensor scope, packed layout and Strata kernel
integration are engineering choices in this patch, not claims of a vendor-approved
KV-cache implementation. This implementation has no measured speedup or
model-quality result yet. See `../NVFP4_VALIDATION.md` for exactly what was run.
