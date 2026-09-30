#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::core {
// Used only from the serving thread on its primary device, never cross-device.
class CheckpointCudaTransport {
    cudaStream_t stream_;
    cudaEvent_t event_ = nullptr;
    uint8_t* stage_ = nullptr;
    size_t capacity_ = 0;
    static bool checked(cudaError_t status, std::string& error) {
        if (status == cudaSuccess) return true;
        error = std::string("checkpoint staging: ") + cudaGetErrorString(status);
        return false;
    }
public:
    explicit CheckpointCudaTransport(cudaStream_t stream) : stream_(stream) {}
    uint8_t* data() noexcept { return stage_; }
    bool reserve(size_t bytes, std::string& error) {
        if (stage_ && capacity_ >= bytes) return true;
        release(); // caller guarantees the previous capture was joined
        if (!checked(cudaHostAlloc(reinterpret_cast<void**>(&stage_), bytes, cudaHostAllocDefault), error)) {
            stage_ = nullptr; cudaGetLastError(); return false;
        }
        if (!checked(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming), error)) {
            cudaGetLastError(); release(); return false;
        }
        capacity_ = bytes; return true;
    }
    bool enqueue(void* dst, const void* src, size_t bytes, std::string& error) {
        return checked(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream_), error);
    }
    bool record(std::string& error) { return checked(cudaEventRecord(event_, stream_), error); }
    bool wait(std::string& error) { return checked(cudaEventSynchronize(event_), error); }
    void drain() noexcept { (void) cudaStreamSynchronize(stream_); }
    void release() noexcept {
        if (stage_) (void) cudaFreeHost(stage_);
        if (event_) (void) cudaEventDestroy(event_);
        stage_ = nullptr; event_ = nullptr; capacity_ = 0;
    }
};
} // namespace strata::core
