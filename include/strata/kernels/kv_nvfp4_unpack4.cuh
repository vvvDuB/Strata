// Exact four-nibble E2M1*2 expansion; no floating arithmetic or codec change.
#pragma once
#include "strata/kernels/kv_nvfp4_codec.hpp"
#if defined(__CUDACC__) || defined(__HIPCC__)
#define STRATA_NV4_UNPACK_HD __host__ __device__ __forceinline__
#else
#define STRATA_NV4_UNPACK_HD inline
#endif
namespace strata::kernels::nvfp4 {
STRATA_NV4_UNPACK_HD uint32_t unpack_twice_e2m1_4(uint32_t x) {
#if defined(__CUDA_ARCH__)
    // PRMT maps four 3-bit magnitudes to bytes; the fourth bit is the sign.
    const uint32_t magnitude = __byte_perm(0x03020100u, 0x0c080604u, x & 0x7777u);
    const uint32_t signs = ((x << 4) & 0x00000080u) | ((x << 8) & 0x00008000u) |
                           ((x << 12) & 0x00800000u) | ((x << 16) & 0x80000000u);
    const uint32_t mask = (signs >> 7) * 255u;
    return (magnitude & ~mask) | (__vsub4(0u, magnitude) & mask);
#else
    // Portable reference, including signed zero's canonical integer zero.
    uint32_t result = 0;
    for (int j = 0; j < 4; ++j)
        result |= uint32_t(uint8_t(twice_e2m1(uint8_t(x >> (4*j))))) << (8*j);
    return result;
#endif
}
}
#undef STRATA_NV4_UNPACK_HD
