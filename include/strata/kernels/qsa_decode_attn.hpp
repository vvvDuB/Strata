// include/strata/kernels/qsa_decode_attn.hpp - plan v0.3 P3/P7: split-K decode attention over the selected cells.
//
// The first attention kernel runs one block per query head (24 blocks on a 48-SM part), reads each key row with one
// thread (uncoalesced) and walks every selected cell serially for the values; at a 4K context (2,051 selected
// cells) that costs ~0.3-0.5 ms per QSA layer.  This one reads the KV POOLS directly through the page table and the
// selection ids (no gather copy), in chunks of CHUNK cells per block, and serves all `n_head / n_head_kv` query
// heads that share a KV head from one read of the chunk:
//
//     grid (ceil(cap / CHUNK), n_head_kv):  s = q.k * scale for 12 heads x CHUNK cells, chunk max m and sum l,
//                                           acc = p . V  ->  partials
//     grid (n_head):                        merge the chunks with the usual log-sum-exp rescale
//
// FP16 pools or INT8 pools (codes + fp16 scale per 64 values, `kv_q8.hpp`).  The grid is sized by the capacity;
// the real count comes from `step[kStepWidth]` as for every other capturable QSA kernel.
#pragma once

#include "strata/kernels/qsa.hpp"

#include <cstdint>

namespace strata::kernels {

struct QsaAttnPools {
    const uint16_t* k_pool = nullptr;   ///< fp16 [page][kv_head][page_size][head_dim], or null when int8
    const uint16_t* v_pool = nullptr;
    const int8_t* k_q = nullptr;        ///< int8 codes, same layout
    const int8_t* v_q = nullptr;
    const uint16_t* k_scale = nullptr;  ///< fp16 [page][kv_head][page_size][head_dim / 64]
    const uint16_t* v_scale = nullptr;
    const uint8_t* k_q4 = nullptr;      ///< q4_0 block_q4_0 [page][kv_head][page_size][head_dim / 32 * 18]
    const uint8_t* v_q4 = nullptr;
    const int32_t* page_table = nullptr;
    const uint8_t* k_nvfp4 = nullptr;   ///< nvfp4::Row, separate from Q4_0
    const uint8_t* v_nvfp4 = nullptr;
    int64_t nvfp4_max_cells = 0;        ///< explicit bounds for sparse NVFP4 loads
    int64_t nvfp4_pages = 0;
};

// Mixed pools are rejected: layout confusion otherwise looks like plausible
// attention, rather than an obvious memory fault.
inline bool qsa_nvfp4_pools_valid(const QsaAttnPools& p, const QsaShapes& s) {
    return p.k_nvfp4 && p.v_nvfp4 && p.k_nvfp4 != p.v_nvfp4 && p.page_table &&
           !p.k_pool && !p.v_pool && !p.k_q && !p.v_q && !p.k_scale && !p.v_scale && !p.k_q4 && !p.v_q4 &&
           s.head_dim == 256 && s.n_head_kv > 0 && s.n_head_kv <= 65535 &&
           s.n_head == 12*s.n_head_kv && s.page_size > 0 && s.page_size <= INT32_MAX && p.nvfp4_max_cells > 0 && p.nvfp4_max_cells <= INT32_MAX &&
           p.nvfp4_pages >= (p.nvfp4_max_cells + s.page_size - 1) / s.page_size;
}

/// Scratch floats for `cap` selected cells: partial accumulators, maxima and sums.
uint64_t qsa_decode_attn_scratch_floats(int64_t cap, const QsaShapes& s);

void qsa_decode_attn_step(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* step,
                          int64_t cap, const QsaShapes& s, float* scratch, float* attn, void* stream);

/// Plan v0.3 P5: `n_q` queries at once, each with its own selection: q [n_q, n_head, 256], ids [n_q, cap], steps
/// [n_q, kStepCount], attn [n_q, n_head, 256]; scratch is `n_q` times the single-query size.
void qsa_decode_attn_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                           int64_t cap, const QsaShapes& s, float* scratch, float* attn, int64_t n_q, void* stream);

}  // namespace strata::kernels
