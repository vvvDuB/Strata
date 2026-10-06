#include "strata/kernels/kv_nvfp4.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include "strata/kernels/kv_nvfp4_convert.cuh"

namespace strata::kernels {
namespace {
void fail(const char* what) { std::fprintf(stderr, "NVFP4 KV: %s\n", what); std::exit(1); }
void check() { const auto e = cudaGetLastError(); if (e != cudaSuccess) fail(cudaGetErrorString(e)); }
void geometry(int64_t cells, const QsaShapes& s) {
    if (s.head_dim != 256 || s.n_head_kv <= 0 || s.n_head_kv > 65535 ||
        s.page_size <= 0 || s.page_size > INT_MAX || cells <= 0 || cells > INT_MAX)
        fail("requires head_dim=256, positive resident geometry, and int32 cell ids");
}
// One complete warp owns one head. No shared memory or block barrier; all
// shuffle participants are live, including the last partially populated CTA.
__device__ __forceinline__ void rht(float (&x)[8], int lane, bool inverse) {
#pragma unroll
    for (int i = 0; i < 8; ++i)
        x[i] *= (inverse ? 1.0f : nvfp4::sign(lane + 32*i)) * (1.0f / 16.0f);
#pragma unroll
    for (int h = 1; h < 32; h <<= 1) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float y = __shfl_xor_sync(0xffffffffu, x[i], h);
            x[i] = (lane & h) ? y - x[i] : x[i] + y;
        }
    }
#pragma unroll
    for (int h = 1; h < 8; h <<= 1) {
#pragma unroll
        for (int b = 0; b < 8; b += 2*h) {
#pragma unroll
            for (int j = 0; j < h; ++j) {
                const float a = x[b+j], c = x[b+j+h];
                x[b+j] = a+c; x[b+j+h] = a-c;
            }
        }
    }
    if (inverse) {
#pragma unroll
        for (int i = 0; i < 8; ++i) x[i] *= nvfp4::sign(lane + 32*i);
    }
}
__device__ __forceinline__ void encode_warp(nvfp4::Row* row, float (&x)[8], int lane) {
    float amax = 0.0f;
    int bad = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) { amax = fmaxf(amax, fabsf(x[i])); bad |= !nvfp4::finite(x[i]); }
#pragma unroll
    for (int o = 16; o; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        bad |= __shfl_xor_sync(0xffffffffu, bad, o);
    }
    const float gs = nvfp4::global_scale(amax);
    if (lane == 0) row->tensor_scale = bad ? nvfp4::from_bits(0x7fc00000u) : gs;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        float bm = fabsf(x[i]);
#pragma unroll
        for (int o = 8; o; o >>= 1) bm = fmaxf(bm, __shfl_xor_sync(0xffffffffu, bm, o, 16));
        const uint8_t sc = (!bad && gs > 0.0f) ? nvfp4::e4m3_encode_scale((bm / gs) / 6.0f) : 0;
        const float sf = nvfp4::e4m3_decode(sc);
        if ((lane & 15) == 0) row->scales[2*i + lane/16] = sc;
        const float a = (!bad && sf > 0.0f) ? (x[i] / gs) / sf : 0.0f;
        const float b = __shfl_down_sync(0xffffffffu, a, 1);
        if ((lane & 1) == 0) row->codes[(32*i + lane)/2] = nvfp4::pack_pair_device(a, b);
    }
}
__global__ void rotate_kernel(const float* src, float* dst, int64_t rows, bool inverse) {
    const int64_t r = int64_t(blockIdx.x)*blockDim.y + threadIdx.y;
    if (r >= rows) return; // warp-uniform
    const int lane = threadIdx.x;
    float x[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) x[i] = src[r*256 + 32*i + lane];
    rht(x, lane, inverse);
#pragma unroll
    for (int i = 0; i < 8; ++i) dst[r*256 + 32*i + lane] = x[i];
}
// One warp per (token, head, K/V). Both sides have one writer only. pos is
// read from DEVICE step in decode, so captured graphs never bake in a token.
__global__ void append_kernel(nvfp4::Row* k, nvfp4::Row* v, const int32_t* table, const int32_t* step,
                              const float* K, const float* V, int64_t p0, int64_t T,
                              int64_t cells, int heads, int page_size) {
    const int64_t task = int64_t(blockIdx.x)*blockDim.y + threadIdx.y;
    if (task >= T*heads*2) return;
    const bool value = (task & 1) != 0;
    const int h = int((task/2) % heads);
    const int64_t token = task/(2*heads);
    const int64_t pos = (step ? int64_t(__ldg(step+kStepPos)) : p0) + token;
    if (pos < 0 || pos >= cells) { __trap(); return; }
    const int64_t page = table[pos/page_size];
    if (page < 0 || page >= (cells + page_size - 1)/page_size) { __trap(); return; }
    const int lane = threadIdx.x;
    float x[8];
    const float* src = (value ? V : K) + (token*heads + h)*256;
#pragma unroll
    for (int i = 0; i < 8; ++i) x[i] = src[32*i+lane];
    rht(x, lane, false);
    encode_warp((value ? v : k) + (page*heads+h)*page_size + pos%page_size, x, lane);
}
__global__ void gather_kernel(const nvfp4::Row* k, const nvfp4::Row* v, const int32_t* table,
                              const int32_t* ids, const int32_t* step, int cap, int64_t cells,
                              int heads, int page_size, uint16_t* K, uint16_t* V) {
    const int64_t task = int64_t(blockIdx.x)*blockDim.y + threadIdx.y;
    const int n = max(0, min(cap, __ldg(step+kStepWidth)));
    if (task >= int64_t(n)*heads*2) return;
    const bool value = (task & 1) != 0;
    const int h = int((task/2) % heads), idx = int(task/(2*heads));
    const int cell = ids[idx];
    int64_t page = -1;
    if (cell >= 0 && cell < cells) page = table[cell/page_size];
    const bool valid = page >= 0 && page < (cells+page_size-1)/page_size;
    const nvfp4::Row* row = valid ? (value ? v : k) + (page*heads+h)*page_size + cell%page_size : nullptr;
    uint16_t* dst = (value ? V : K) + (int64_t(idx)*heads+h)*256;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int d = threadIdx.x+32*i;
        dst[d] = __half_as_ushort(__float2half_rn(row ? nvfp4::decode(*row, d) : 0.0f));
    }
}
unsigned grid_rows(int64_t rows) {
    if (rows < 0 || rows > int64_t(INT_MAX)*4) fail("grid extent overflow");
    return unsigned((rows+3)/4);
}
} // namespace
void nvfp4_rotate_cuda(const float* src, float* dst, int64_t rows, bool inverse, void* stream) {
    if (rows == 0) return;
    if (!src || !dst || rows < 0) fail("invalid rotation buffers/extent");
    rotate_kernel<<<grid_rows(rows), dim3(32,4), 0, (cudaStream_t)stream>>>(src,dst,rows,inverse); check();
}
void kv_append_nvfp4_step(uint8_t* k, uint8_t* v, const int32_t* table, const int32_t* step,
                         const float* K, const float* V, int64_t cells, const QsaShapes& s, void* stream) {
    geometry(cells,s);
    if (!k || !v || k == v || !K || !V || !table || !step) fail("invalid append buffers");
    append_kernel<<<grid_rows(s.n_head_kv*2), dim3(32,4), 0, (cudaStream_t)stream>>>(
        (nvfp4::Row*)k,(nvfp4::Row*)v,table,step,K,V,0,1,cells,int(s.n_head_kv),int(s.page_size)); check();
}
void kv_append_nvfp4(uint8_t* k, uint8_t* v, const int32_t* table, int64_t p0, int64_t T,
                    const float* K, const float* V, int64_t cells, const QsaShapes& s, void* stream) {
    geometry(cells,s);
    if (T == 0) return;
    if (!k || !v || k == v || !K || !V || !table || p0 < 0 || T < 0 || p0 > cells || T > cells-p0)
        fail("invalid batch append buffers/extent");
    append_kernel<<<grid_rows(T*s.n_head_kv*2), dim3(32,4), 0, (cudaStream_t)stream>>>(
        (nvfp4::Row*)k,(nvfp4::Row*)v,table,nullptr,K,V,p0,T,cells,int(s.n_head_kv),int(s.page_size)); check();
}
void kv_gather_nvfp4_step(const uint8_t* k, const uint8_t* v, const int32_t* table,
                         const int32_t* ids, const int32_t* step, int64_t cap, int64_t cells,
                         const QsaShapes& s, uint16_t* K, uint16_t* V, void* stream) {
    geometry(cells,s);
    if (cap == 0) return;
    if (!k || !v || !table || !ids || !step || !K || !V || K == V || cap < 0 || cap > INT_MAX)
        fail("invalid gather buffers/extent");
    gather_kernel<<<grid_rows(cap*s.n_head_kv*2), dim3(32,4), 0, (cudaStream_t)stream>>>(
        (const nvfp4::Row*)k,(const nvfp4::Row*)v,table,ids,step,int(cap),cells,int(s.n_head_kv),int(s.page_size),K,V); check();
}
} // namespace strata::kernels
