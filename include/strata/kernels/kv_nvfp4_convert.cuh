// Device conversion shared by the production append kernel and GPU parity test.
#pragma once
#include "strata/kernels/kv_nvfp4_codec.hpp"
#include <cuda_runtime.h>
#if !defined(__HIPCC__) && defined(__CUDACC_VER_MAJOR__) && \
    (__CUDACC_VER_MAJOR__ > 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 8))
#include <cuda_fp4.h>
#define STRATA_NV4_NATIVE_CONVERT 1
#else
#define STRATA_NV4_NATIVE_CONVERT 0
#endif
namespace strata::kernels::nvfp4 {
__device__ __forceinline__ uint8_t pack_pair_device(float a, float b) {
#if STRATA_NV4_NATIVE_CONVERT && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    return __nv_cvt_float2_to_fp4x2(make_float2(a,b), __NV_E2M1, cudaRoundNearest);
#else
    return uint8_t(e2m1_encode(a) | (e2m1_encode(b) << 4));
#endif
}
} // namespace strata::kernels::nvfp4
