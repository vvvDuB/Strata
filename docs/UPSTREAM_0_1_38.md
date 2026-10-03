# Owner integration of 0.1.38

This merges official tag `v0.1.38` (`99f3dbd`) into the owner's integrated 0.1.31
(`2beb49b`). Keep both parents: this is not an upstream replacement of the fork.

The owner balanced/root/fork cache, legacy disk writeback and V6 safety fixes,
per-layer prefill cancellation, dynamic Anthropic effort, shell-only server
configuration, CUDA PLE workaround and AVX2 fixes remain. V7 stays excluded.
Upstream 0.1.38 layouts, kernels, frontend/tool/security/lifecycle changes and
the `prompt_read` completion field are integrated with owner phase/cache metrics.

## Q3 profile

The installed model profile is `/home/atef/AI/bin/Qwen3.8-Flash-Next-IQ3_XXS.sh`.
It supplies literal native settings after the Python wrapper's `--` separator:

```text
--prefill 4096
--prefill-stream-min 4096
--prompt-cache 12
--prompt-cache-policy balanced
--prompt-cache-every 4096
--prompt-cache-root 1
--max-context 131072
```

Keep the existing INT8/mmap/MTP settings and 8192 MiB/four-file legacy conversation
store. No Claude-only prefix file, RAM/shared cache tier, external settings JSON,
systemd unit or autostart is introduced. Binding and authentication also stay in
the Bash file. Never publish its API key or private inference/cache artifacts.

`--prefill-stream-min N` takes a strictly positive int64 integer. It configures
the process before any prefill buffers or stage threads are created. Explicit
CLI values override `STRATA_PREFILL_STREAM_MIN`; without the flag, the legacy
benchmark environment and owner default 2048 remain supported. The value is
reported at startup. It is distinct from the prefill chunk and checkpoint interval.
Existing snapshot identity includes inference arguments and executable identity:
a new build/setting can make the first conversation cold. Do not delete old caches.
The legacy store now treats valid snapshots of another identity as safe misses,
not corrupt files: it preserves them for rollback and never indexes/reuses/evicts
them in the new identity. The 8192 MiB/four-entry limit applies to the active
identity; preserved old identities can occupy additional disk space. Remove or
archive them only with the owner's approval. Malformed files and abandoned atomic
temporary files retain their existing cleanup behavior.

## Measurements and limitations

On Ryzen 7 5700X3D, 32 GiB RAM, RTX 5070 12 GiB, Q3 and CUDA 12.8/SM120, the
same-binary 2048/4096 streaming ABBA showed 107.52 -> 156.52 tok/s for 2209-token
inputs and 166.36 -> 370.36 for 3525-token inputs. A single long candidate trial
kept 45059/45060 tokens on disk return and reached 74.51 tok/s warmed decode.
These are not general 2x gains, nor a significant decode improvement. The 45k
cold trial is paging-sensitive. Eight checkpoints gave no decode gain and worse
edit coverage, so the profile keeps twelve.

Full evidence is in the parent workspace's
`Docs/Strata-v0.1.38-Q3-middle-ground-2026-10-03.md`; the deployment report records
the rebuilt binary and installed-profile smoke tests. No complete 128k input,
real Pi/Claude long-task replay, multi-GPU/HIP or full-model logit parity is claimed.

Native parsing regressions: `tools/test_prefill_stream_cli.py` (set
`STRATA_TEST_EXE` when testing a noncanonical build). Startup policy/precedence
regressions: `prefill_stream_policy_test` and `prefill_stream_policy_env_test`.
The server must forward the native flag without consulting JSON settings.
