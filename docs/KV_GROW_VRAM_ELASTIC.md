# Elastic KV and expert cache

The native `--kv-grow` option now works with `--serve --vram-elastic`, including
NVFP4 KV and the resident low-RAM expert complement. Add the flag after the
Python server's `--` separator:

```sh
--kv nvfp4 --kv-resident 0 --kv-grow --max-context 131072 \
--vram-elastic --vram-segment-mib 64
```

It remains opt-in. `STRATA_KV_GROW=1` also enables it; an explicit environment
value overrides the CLI flag. The default initial KV allocation covers 16384
cells, with growth in steps of 8192. Physical allocation rounds to CUDA's
granularity. The reserved addresses cover the entire configured context and
remain fixed, including those used by CUDA graphs.

The combined startup allocation of KV and expert cache is the memory ceiling.
When a prompt, restored conversation or decode window needs more cells, the
engine drains GPU work and pending adaptive swaps, removes residency entries
for the cache tail, releases whole cache segments and maps more KV memory.
The prompt buffers are relaid out inside the remaining slots. A shorter
request trims unused KV after the outgoing conversation has been parked and
refills the cache, up to its startup capacity.

The low-RAM complement remains valid: an expert missing from both GPU and RAM
can still be read from the model files. Refills from reusable file staging
buffers synchronize their copies before another read. Very long requests can
therefore incur more file reads. This does not require copying every expert
into RAM or changing the resident budget.

On-demand vision shares the same cache resize/refill operations. Its lease
records the cache size at `VRAM BEGIN` and restores exactly that size at
`VRAM END`, including when a long conversation already has a larger KV.
It does not restore the larger startup cache over live KV or change the
configured reserve. Generation is refused while a vision lease is open.
A failed KV growth or cache restoration stops the native request loop rather
than continuing with partially changed allocations.

The segmented path currently supports one NVIDIA GPU, a nonempty expert
profile, no streaming KV, no batch slots and no peer/helper GPUs. The existing
unsegmented `--kv-grow` path retains its requirement that every expert be in
RAM. Without a usable cache coordinator, partial KV allocation falls back to
the full context at startup or refuses startup if that does not fit.

`INFO` reports `kv_grow`, `kv_cells` and `kv_mapped_mib` for elastic pools;
the latter is zero without elastic pools. Growth and trim logs report the
current KV capacity and cache slot counts.

After every successful KV grow/trim, the engine sends updated cache slot,
cache memory and KV capacity facts on stdout. The server consumes them in its
output reader while the request is running, so `/metrics` and `/#monitor`
track the current cache rather than its startup size. Vision/manual VRAM
replies also update both the slot count and cache memory.

## Validation

Measured on 2026-10-07 with RTX 5070 12 GiB / SM120, CUDA 13.4,
Ryzen 5700X3D, 32 GiB RAM, IQ3_XXS experts, NVFP4 KV, context 131072,
resident budget 22 GiB, reserve 640 MiB and on-demand GPU vision:

| Warm short-request measurement | Fixed KV | Elastic KV |
| --- | ---: | ---: |
| Startup GPU expert slots | 2835 | 3354 |
| Expert cache | 4.64 GiB | 5.48 GiB |
| Mean generation, six warm 96-token requests | 37.70 tok/s | 41.17 tok/s |

This is approximately 9.2% faster on one Italian binary-search explanation
prompt. Both runs had six earlier warm-up requests. The adaptive cache was
enabled; CPU/GPU rounding and different cache contents can change tokens.
These measurements do not establish a universal speedup.

A real 40045-token prompt grew KV to 40960 cells and reduced the cache from
3354 to 3256 slots. An uncached image then borrowed space down to 2478 slots;
encoder exit restored exactly 3256 slots. The short image request trimmed KV
to 8192 cells and restored all 3354 slots. The image was described as a red
square and blue circle; the cached repeat returned the same description and
reused 218 prompt tokens. Subsequent text arithmetic returned `4`.

A 16033-token prompt followed by 512 generated tokens crossed the next KV
boundary during decoding: KV grew from 16384 to 24576 cells. An invalid PNG
then returned HTTP 400, and the following text request returned `4` after
trimming KV and restoring all cache slots.

`kv_vram_elastic_test` checks three full-capacity grow/trim cycles, temporary
vision leases while KV is grown, a padded final cache segment, live-byte
preservation and replay of captured CUDA graphs. Run it with:

```sh
cmake --build build --target kv_vram_elastic_test vmm_test
ctest --test-dir build -R '^(kv_vram_elastic_test|vmm_test)$' --output-on-failure
```

Both GPU tests passed; compute-sanitizer memcheck reported zero errors for
the combined test. The 16 on-demand vision server tests and the rebuilt
file expert source and profile tests also passed. The production service
was left stopped; the existing launcher received only `--kv-grow`.
