// KV pools and a segmented expert cache must resize together without invalidating captured addresses.
#include "strata/core/expert_cache.hpp"
#include "strata/core/layer.hpp"
#include "strata/kernels/kv_nvfp4.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace c = strata::core;
static void check(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
static void ck(cudaError_t e) { check(e == cudaSuccess, cudaGetErrorString(e)); }

int main() {
    if (cudaFree(nullptr) != cudaSuccess || !c::vmm_available()) return 77;
    try {
        constexpr int64_t MiB = 1LL << 20, capacity = 131072;
        c::ModelGeometry g;
        c::qsa_set_kv_nvfp4(true);
        c::qsa_set_kv_elastic(true, 64);
        void* arena = nullptr;
        ck(cudaMalloc(&arena, c::qsa_state_bytes(g, capacity)));
        c::QsaState state{};
        check(c::qsa_state_init(g, capacity, arena, state) > 0, "elastic NVFP4 state");
        const int64_t initial = c::qsa_kv_elastic_cells();
        check(initial < capacity, "partial initial KV");
        const int64_t initial_kv = c::qsa_kv_elastic_mapped_bytes();
        c::ExpertCache cache;
        cache.set_segment_bytes(64 * MiB);
        std::string err;
        // Deliberately leave allocation-granularity padding after the final logical slot.
        check(cache.open(256, 1, 256, MiB + 16, err), err.c_str());
        const int64_t full = cache.mapped_bytes();
        auto* base = cache.device_slot(0);
        ck(cudaMemset(base, 0x31, cache.full_bytes()));
        ck(cudaMemset(state.k_nvfp4, 0x72, 256));
        ck(cudaDeviceSynchronize());
        cudaStream_t stream;
        cudaGraph_t graph;
        cudaGraphExec_t exec;
        void* output = nullptr;
        ck(cudaMalloc(&output, 512));
        ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        ck(cudaMemcpyAsync(output, base, 256, cudaMemcpyDeviceToDevice, stream));
        ck(cudaMemcpyAsync((uint8_t*) output + 256, state.k_nvfp4, 256, cudaMemcpyDeviceToDevice, stream));
        ck(cudaStreamEndCapture(stream, &graph));
        ck(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        auto replay = [&] {
            ck(cudaGraphLaunch(exec, stream));
            ck(cudaStreamSynchronize(stream));
            std::vector<uint8_t> bytes(512);
            ck(cudaMemcpy(bytes.data(), output, bytes.size(), cudaMemcpyDeviceToHost));
            check(std::all_of(bytes.begin(), bytes.begin()+256, [](auto b) { return b == 0x31; }), "cache prefix preserved");
            check(std::all_of(bytes.begin()+256, bytes.end(), [](auto b) { return b == 0x72; }), "KV prefix preserved");
        };
        for (int cycle = 0; cycle < 3; ++cycle) {
            const int64_t need = c::qsa_kv_elastic_need(capacity) * c::vmm_granularity();
            const int64_t keep = (cache.mapped_bytes() - need) / cache.segment_bytes() * cache.segment_bytes();
            check(keep > 0 && cache.shrink(keep, err), "give cache memory to KV");
            check(c::qsa_kv_elastic_grow(capacity, [] { return c::VmmChunk(0); }), "grow full KV");
            check(c::qsa_kv_elastic_cells() == capacity, "full context still fits");
            check(cache.mapped_bytes() + (int64_t) c::qsa_kv_elastic_mapped_bytes() <= full + initial_kv,
                  "combined physical budget");
            replay();
            // The encoder borrows space while a long conversation still holds its KV. END must restore this
            // smaller cache exactly, never the startup size which would compete with the grown KV.
            const auto before_vision = cache.mapped_bytes();
            check(cache.shrink(before_vision - 64 * MiB, err), "vision lease shrink");
            replay();
            check(cache.grow(before_vision, err) && cache.mapped_bytes() == before_vision, "exact vision lease restore");
            replay();
            check(c::qsa_kv_elastic_shrink(64, c::vmm_chunk_free) > 0, "short conversation trims KV");
            check(c::qsa_kv_elastic_cells() == initial, "initial KV restored");
            check(cache.grow(cache.full_bytes(), err), "restore final padded segment");
            check(cache.slots() == cache.full_slots() && cache.mapped_bytes() == full, "all cache slots restored");
            replay();
            uint8_t last = 0;
            ck(cudaMemset(cache.device_slot(255), 0x55, MiB+16));
            ck(cudaMemcpy(&last, cache.device_slot(255)+MiB+15, 1, cudaMemcpyDeviceToHost));
            check(last == 0x55, "last slot is writable after restore");
        }
        ck(cudaGraphExecDestroy(exec)); ck(cudaGraphDestroy(graph)); ck(cudaStreamDestroy(stream));
        ck(cudaFree(output)); ck(cudaFreeHost(state.host_step)); ck(cudaFreeHost(state.host_pos)); ck(cudaFree(arena));
        std::puts("KV + VRAM elastic: growth, trim, vision leases, full restoration and captured graphs passed");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "KV + VRAM elastic: %s\n", e.what());
        return 1;
    }
}
