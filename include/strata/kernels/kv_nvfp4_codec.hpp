// Portable wire format and bit-exact scalar reference for RHT256 + NVFP4 KV.
// Each token/head is an independent NVFP4 tensor: E2M1 payload, one E4M3
// scale per 16 values, one FP32 tensor scale. This is NOT a Q4_0 codebook.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cfloat>
#include <limits>
#include <type_traits>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define STRATA_NV4_HD __host__ __device__ __forceinline__
#else
#define STRATA_NV4_HD inline
#endif
namespace strata::kernels::nvfp4 {
inline constexpr int head_dim = 256, block_size = 16, groups = 16;
inline constexpr int snapshot_format = 5; // 4 reserved for the earlier experimental TQ4-V format
inline constexpr uint32_t rotation_seed = 0xa341316cu;
struct alignas(4) Row {
    uint8_t codes[128]; // adjacent elements: even in low nibble, odd in high nibble
    uint8_t scales[16]; // nonnegative E4M3, NOT E8M0 or FP16
    float tensor_scale;
};
static_assert(sizeof(Row) == 148 && alignof(Row) == 4);
static_assert(offsetof(Row, scales) == 128 && offsetof(Row, tensor_scale) == 144);
static_assert(std::is_trivially_copyable_v<Row>);

STRATA_NV4_HD uint32_t bits(float x) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return __float_as_uint(x);
#else
    uint32_t u; std::memcpy(&u, &x, 4); return u;
#endif
}
STRATA_NV4_HD float from_bits(uint32_t u) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return __uint_as_float(u);
#else
    float x; std::memcpy(&x, &u, 4); return x;
#endif
}
STRATA_NV4_HD bool finite(float x) { return (bits(x) & 0x7f800000u) != 0x7f800000u; }
STRATA_NV4_HD float magnitude(float x) { return from_bits(bits(x) & 0x7fffffffu); }
STRATA_NV4_HD float sign(int d) {
    // A fixed Rademacher diagonal, part of the serialized format. Do not change
    // the seed/hash without changing snapshot_format and the disk identity.
    uint32_t u = uint32_t(d) + rotation_seed;
    u = (u ^ (u >> 16)) * 0x7feb352du;
    u = (u ^ (u >> 15)) * 0x846ca68bu;
    u ^= u >> 16;
    return (u & 1u) ? -1.0f : 1.0f;
}
// Exact signed integer equal to 2 * E2M1. All codes are exactly representable
// by FP16/INT8, letting the prompt MMA apply the actual scales in FP32.
STRATA_NV4_HD int twice_e2m1(uint8_t q) {
    const int a = int((0x0c08060403020100ull >> (8 * (q & 7))) & 255u);
    return (q & 8) ? -a : a;
}
STRATA_NV4_HD float e2m1_decode(uint8_t q) {
    const float a = 0.5f * float(twice_e2m1(q & 7));
    return (q & 8) ? -a : a;
}
STRATA_NV4_HD uint8_t e2m1_encode(float x) {
    const uint32_t u = bits(x);
    const float a = magnitude(x);
    // CUDA SATFINITE semantics: NaN -> positive MAXNORM; infinities saturate.
    if ((u & 0x7fffffffu) > 0x7f800000u) return 7;
    int q;
    if (a <= 0.25f) q = 0;
    else if (a < 0.75f) q = 1;
    else if (a <= 1.25f) q = 2;
    else if (a < 1.75f) q = 3;
    else if (a <= 2.5f) q = 4;
    else if (a < 3.5f) q = 5;
    else if (a <= 5.0f) q = 6;
    else q = 7;
    return uint8_t(q | ((u >> 28) & 8u)); // preserve signed zero
}
STRATA_NV4_HD float e4m3_decode(uint8_t q) {
    const int e = (q >> 3) & 15, m = q & 7;
    float a = e == 0 ? float(m) * (1.0f / 512.0f)
                    : from_bits((uint32_t(e + 120) << 23) | (uint32_t(m) << 20));
    if ((q & 127) == 127) a = from_bits(0x7fc00000u);
    return (q & 128) ? -a : a;
}
// Nonnegative E4M3 round-to-nearest-even, saturating at 448. Scale callers
// supply finite nonnegative inputs; zero is canonical +0.
STRATA_NV4_HD uint8_t e4m3_encode_scale(float x) {
    if (!(x > 0.0f)) return 0;
    if (x >= 448.0f) return 126;
    if (x < 0.015625f) {
        const float y = x * 512.0f;
        int n = int(y);
        const float f = y - float(n);
        n += (f > 0.5f || (f == 0.5f && (n & 1)));
        return uint8_t(n);
    }
    const uint32_t u = bits(x);
    int e = int((u >> 23) & 255) - 127 + 7;
    int m = int((u >> 20) & 7);
    const uint32_t rem = u & 0xfffffu;
    m += (rem > 0x80000u || (rem == 0x80000u && (m & 1)));
    if (m == 8) { m = 0; ++e; }
    return uint8_t((e << 3) | m);
}
STRATA_NV4_HD float decode(const Row& r, int d) {
    const uint8_t q = uint8_t(r.codes[d / 2] >> ((d & 1) * 4));
    return e2m1_decode(q) * (e4m3_decode(r.scales[d / 16]) * r.tensor_scale);
}
STRATA_NV4_HD float global_scale(float amax) {
    if (amax == 0.0f) return 0.0f;
    const float s = amax / 2688.0f; // 6 * 448
    return s < FLT_MIN ? FLT_MIN : s; // keep scale normal even on FTZ devices
}

// Host reference. Forward R = H D, inverse R^T = D H; R is not self-inverse.
inline void rotate(float* x, bool inverse = false) {
    for (int d = 0; d < head_dim; ++d) x[d] *= (inverse ? 1.0f : sign(d)) * (1.0f / 16.0f);
    for (int h = 1; h < head_dim; h *= 2)
        for (int base = 0; base < head_dim; base += 2 * h)
            for (int j = 0; j < h; ++j) {
                const float a = x[base+j], b = x[base+j+h];
                x[base+j] = a+b; x[base+j+h] = a-b;
            }
    if (inverse) for (int d = 0; d < head_dim; ++d) x[d] *= sign(d);
}
inline Row encode_rotated(const float* x) {
    Row r{};
    float amax = 0;
    for (int d = 0; d < head_dim; ++d) {
        if (!finite(x[d])) {
            // Propagate invalid model activations, never silently replace by a
            // plausible zero row. Attention output will remain non-finite.
            r.tensor_scale = std::numeric_limits<float>::quiet_NaN(); return r;
        }
        if (magnitude(x[d]) > amax) amax = magnitude(x[d]);
    }
    r.tensor_scale = global_scale(amax);
    if (amax == 0) return r;
    for (int b = 0; b < groups; ++b) {
        float bm = 0;
        for (int j = 0; j < block_size; ++j)
            if (magnitude(x[b*16+j]) > bm) bm = magnitude(x[b*16+j]);
        r.scales[b] = e4m3_encode_scale((bm / r.tensor_scale) / 6.0f);
        const float sb = e4m3_decode(r.scales[b]);
        for (int j = 0; j < 16; j += 2) {
            const int d = b*16+j;
            const float a = sb > 0 ? (x[d] / r.tensor_scale) / sb : 0;
            const float c = sb > 0 ? (x[d+1] / r.tensor_scale) / sb : 0;
            r.codes[d/2] = uint8_t(e2m1_encode(a) | (e2m1_encode(c) << 4));
        }
    }
    return r;
}
inline Row encode(const float* x) {
    float r[head_dim]; std::memcpy(r, x, sizeof(r)); rotate(r); return encode_rotated(r);
}
inline void decode_row(const Row& r, float* out, bool inverse = true) {
    for (int d = 0; d < head_dim; ++d) out[d] = decode(r, d);
    if (inverse) rotate(out, true);
}
} // namespace strata::kernels::nvfp4
#undef STRATA_NV4_HD
