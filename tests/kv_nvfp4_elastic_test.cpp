// NVFP4 pools must survive VMM growth/shrink, preserve live bytes, and clear new cells.
#include "strata/core/layer.hpp"
#include "strata/core/vmm.hpp"
#include "strata/kernels/kv_nvfp4.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace c = strata::core;
namespace k = strata::kernels;
static void check(bool ok, const char* why) {
    if (!ok) throw std::runtime_error(why);
}
static void ck(cudaError_t result) { check(result == cudaSuccess, cudaGetErrorString(result)); }

int main() {
    if (cudaFree(nullptr) != cudaSuccess || !c::vmm_available()) {
        std::puts("NVFP4 elastic pools: CUDA VMM unavailable");
        return 77;
    }
    try {
        c::ModelGeometry geometry;
        constexpr int64_t capacity = 32768;
        c::qsa_set_kv_nvfp4(true);
        c::qsa_set_kv_elastic(true, 64);
        void* arena = nullptr;
        const auto arena_bytes = c::qsa_state_bytes(geometry, capacity);
        ck(cudaMalloc(&arena, arena_bytes));
        c::QsaState state{};
        const auto used = c::qsa_state_init(geometry, capacity, arena, state);
        check(used > 0 && used <= arena_bytes, "state arena extent");
        check(state.kv_nvfp4 && state.kv_elastic >= 0 && state.k_nvfp4 && state.v_nvfp4,
              "NVFP4 pointers in the elastic range");
        const auto initial_cells = c::qsa_kv_elastic_cells();
        check(initial_cells >= 64 && initial_cells < capacity, "partial initial mapping");
        const size_t row_bytes = geometry.n_head_kv * k::kv_nvfp4_bytes_per_head();
        const size_t initial_bytes = initial_cells * row_bytes;
        ck(cudaMemset(state.k_nvfp4, 0x3a, initial_bytes));
        ck(cudaMemset(state.v_nvfp4, 0x7b, initial_bytes));
        ck(cudaDeviceSynchronize());
        check(c::qsa_kv_elastic_need(capacity) > 0, "growth requires chunks");
        check(c::qsa_kv_elastic_grow(capacity, [] { return c::VmmChunk(0); }), "grow");
        check(c::qsa_kv_elastic_cells() == capacity, "full mapping");
        std::vector<unsigned char> bytes(capacity * row_bytes);
        for (auto entry : {std::pair{state.k_nvfp4, 0x3a}, std::pair{state.v_nvfp4, 0x7b}}) {
            ck(cudaMemcpy(bytes.data(), entry.first, bytes.size(), cudaMemcpyDeviceToHost));
            check(std::all_of(bytes.begin(), bytes.begin() + initial_bytes,
                              [&](auto value) { return value == entry.second; }), "live bytes preserved");
            check(std::all_of(bytes.begin() + initial_bytes, bytes.end(),
                              [](auto value) { return value == 0; }), "new cells cleared");
        }
        ck(cudaDeviceSynchronize());
        check(c::qsa_kv_elastic_shrink(64, c::vmm_chunk_free) > 0, "shrink");
        check(c::qsa_kv_elastic_cells() == initial_cells, "initial capacity restored");
        c::qsa_state_zero(state, geometry, nullptr);
        ck(cudaDeviceSynchronize());
        for (auto pointer : {state.k_nvfp4, state.v_nvfp4}) {
            ck(cudaMemcpy(bytes.data(), pointer, initial_bytes, cudaMemcpyDeviceToHost));
            check(std::all_of(bytes.begin(), bytes.begin() + initial_bytes,
                              [](auto value) { return value == 0; }), "reset clears mapped cells");
        }
        check(c::qsa_kv_elastic_grow(capacity, [] { return c::VmmChunk(0); }), "regrow");
        for (auto pointer : {state.k_nvfp4, state.v_nvfp4}) {
            ck(cudaMemcpy(bytes.data(), pointer, bytes.size(), cudaMemcpyDeviceToHost));
            check(std::all_of(bytes.begin(), bytes.end(), [](auto value) { return value == 0; }),
                  "regrown cells cleared");
        }
        ck(cudaFreeHost(state.host_step));
        ck(cudaFreeHost(state.host_pos));
        ck(cudaFree(arena));
        std::puts("NVFP4 elastic pools: growth, shrink, preservation and clearing passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "NVFP4 elastic pools: %s\n", error.what());
        return 1;
    }
}
