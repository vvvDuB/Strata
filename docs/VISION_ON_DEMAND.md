# Vision on demand

`vision.mode: "on-demand"` starts a fresh `strata-vision --once` process only
for an image that has no cached embeddings. It loads the projector, encodes
one image, writes the existing SVE1 record, and exits. The server reaps the
process before returning embeddings or restoring the expert cache. No vision
process, CUDA context, projector weights, pinned buffers or encoder work
buffers remain during text inference. Cached embeddings are bounded to 64
files on disk, rather than a persistent `/tmp` RAM filesystem cache.

The default remains `resident` for existing installations. OpenAI image parts,
Anthropic image blocks, tool images, image token positions and SVE1 embeddings
use the existing multimodal path. The decoder's existing M-RoPE position table
is still required by `--vision`; this mode removes the encoder's residency.

For a direct launch, add these server options before the native `--` separator:

```sh
--vision-exe /path/to/strata-vision \
--vision-mmproj /path/to/mmproj.gguf \
--vision-model /path/to/text-model-00001-of-00002.gguf \
--vision-mode on-demand --vision-gpu --vision-max-tokens 1024
```

After the separator, enable the existing image protocol and segmented cache:

```sh
--vision --vram-elastic --vram-segment-mib 64
```

No permanent encoder reserve is added. `--vision-vram-mib` sets the free VRAM
required temporarily by the encoder (default 2048 MiB); it does not change
`--vram-reserve-mib`. Choose it for the projector, backend and image token cap.
The CPU mode omits `--vision-gpu` and does not borrow GPU memory. GPU cache
borrowing currently supports one NVIDIA GPU, including the resident low-RAM
expert complement. An insufficient cache floor refuses the image before
starting the encoder, then restores the cache.

Setup accepts `--vision gpu --vision-mode on-demand` (or `--vision cpu`). This
mode builds locally so the encoder has `--once` and the engine has the temporary
lease protocol. Legacy JSON configs can set the same values in `vision`:
`mode`, `gpu`, `max_tokens`, `vram_mib`, `timeout_s` (default 120 seconds) and
`cache_dir` (a disk directory without spaces). GPU configs also set
`vram_elastic: true`, `vram_segment_mib: 64`, and retain native `--vision`.

While encoding, new text admissions wait and existing batch requests finish.
Image decoding also runs alone because the decoder's M-RoPE table is shared;
text-only batch requests still run concurrently.
Only missing free memory is taken from the end of the expert cache, rounded
to a segment. `VRAM BEGIN <MiB>` records its preceding size; `VRAM END` restores
those slots and their experts, including file-backed experts absent from the
RAM complement. Refilling synchronizes reusable file staging buffers before
another read can overwrite them. This implementation refills immediately;
lazy refilling is not required to release all encoder memory. A failed restore
unloads the engine rather than resuming text with an incomplete low-RAM cache.

Timeout, bad image, process failure and malformed embeddings all reap the
child and restore the lease. Native prompt image rows are freed after each
request. Decoder CUDA graphs and conversation caches keep their existing
lifecycle and can still grow during ordinary inference.

Measured on 2026-10-07, RTX 5070 SM120, CUDA 13.4, Ryzen 5700X3D and the BF16
Qwen3.8-Flash-Next projector (907,543,008 bytes):

| Input | Image tokens | Load, encode and exit | Additional peak VRAM | Peak child RSS |
| --- | ---: | ---: | ---: | ---: |
| 448 × 448 PNG | 196 | 1.18 s | 1190 MiB | 694 MiB |
| 2048 × 2048 PNG | 1024 | 1.36 s | 1314 MiB | 705 MiB |

In both measurements the process disappeared and VRAM returned exactly to
the preceding value. With the real IQ3 model, 22 GiB resident expert budget,
NVFP4 KV, 131072 context and the existing 640 MiB reserve, the encoder worked
while only 134–198 MiB were free before borrowing. A temporary 1408 MiB
budget sufficed, and all 2835 expert slots and the 8192-token prompt chunk
were restored after every encode. Cache refill took about 0.6 s in these
tests. The model described a red square and a blue circle correctly; resending
the same picture started no encoder. Text before and after returned the same
answer. These times are measurements for this hardware and token cap, not
general guarantees for other projectors.

For a repeated 64-token text request, four warm runs averaged 37.45 tok/s with
the preceding text-only release and 37.15 tok/s with on-demand vision enabled
(a 0.8% difference). GPU/CPU expert rounding and adaptive cache changes already
vary the completions between repeated requests in the preceding release;
this check does not claim bit-for-bit determinism across cache configurations.

`python -m unittest serve.test_vision_on_demand` covers child lifetime,
timeouts, crash/error cleanup, embedding validation, cache hits, failed
borrowing/restoration and exclusion against two concurrent batch requests.
