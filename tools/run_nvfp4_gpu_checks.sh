#!/usr/bin/env bash
# Run from any directory. Builds nothing, never downloads weights, never treats
# an unavailable GPU or an unexercised native conversion path (exit 77) as a pass.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build=${1:-"$root/build-nvfp4"}
logdir=${2:-"$build/nvfp4-check-logs"}
mkdir -p -- "$logdir"
for test in kv_nvfp4_native_conversion kv_nvfp4_gpu_test kv_nvfp4_unpack_test; do
    executable="$build/$test"
    if [[ ! -x "$executable" ]]; then
        printf 'Missing executable: %s\nBuild with -DSTRATA_TEST_NVFP4=ON first.\n' "$executable" >&2
        exit 2
    fi
    "$executable" 2>&1 | tee "$logdir/$test.log"
done
if ! command -v compute-sanitizer >/dev/null 2>&1; then
    printf 'compute-sanitizer missing: device sanitizer gates have NOT passed.\n' >&2
    exit 2
fi
# Both executables, including the native conversion used by production.
for test in kv_nvfp4_native_conversion kv_nvfp4_gpu_test kv_nvfp4_unpack_test; do
    for tool in memcheck racecheck initcheck synccheck; do
        compute-sanitizer --tool "$tool" --error-exitcode 99 "$build/$test" \
            2>&1 | tee "$logdir/$test-$tool.log"
    done
done
printf 'Device parity and sanitizer gates passed. Model quality/throughput still require separate measurement.\n'
