// Resident K+V NVFP4 cache with fused randomized Hadamard encode.
// Main QSA layers only. The MTP drafter retains INT8. See docs/NVFP4_KV.md.
#pragma once
#include "strata/kernels/kv_nvfp4_codec.hpp"
#include "strata/kernels/qsa.hpp"
namespace strata::kernels {
inline constexpr uint64_t kv_nvfp4_bytes_per_head() { return sizeof(nvfp4::Row); }
inline uint64_t kv_nvfp4_bytes_per_cell(const QsaShapes& s) {
    return uint64_t(s.n_head_kv) * 2 * sizeof(nvfp4::Row);
}
// Out-of-place and exactly in-place allowed. Partial overlap is not supported.
// Forward H*D for queries (after RoPE), inverse D*H for attention outputs.
void nvfp4_rotate_cuda(const float* src, float* dst, int64_t rows, bool inverse, void* stream);
// Input K is already normalized and RoPE-rotated; V is unrotated. Both are
// transformed in registers and encoded WITHOUT modifying the input buffers.
void kv_append_nvfp4_step(uint8_t* k, uint8_t* v, const int32_t* table, const int32_t* step,
                         const float* K, const float* V, int64_t max_cells, const QsaShapes& s, void* stream);
void kv_append_nvfp4(uint8_t* k, uint8_t* v, const int32_t* table, int64_t pos0, int64_t T,
                    const float* K, const float* V, int64_t max_cells, const QsaShapes& s, void* stream);
// Dequantize only the selected rows, still in the rotated basis. Invalid ids
// or nonresident pages write zeros without reading outside either pool.
void kv_gather_nvfp4_step(const uint8_t* k, const uint8_t* v, const int32_t* table,
                         const int32_t* ids, const int32_t* step, int64_t cap, int64_t max_cells,
                         const QsaShapes& s, uint16_t* K, uint16_t* V, void* stream);
} // namespace strata::kernels
