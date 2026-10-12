// TEST ONLY. Host memory-transfer fixture for the actual cache serialization code.
// Never add this directory to production targets. It is NOT a CUDA emulator and
// cannot validate kernels, CUDA graphs, memory ordering, or device performance.
#pragma once
#include <cstddef>
#include <cstring>
#include <cstdlib>
using cudaStream_t = void*;
using cudaGraphExec_t = void*;
using cudaGraph_t = void*;
using cudaEvent_t = void*;
enum cudaError_t {cudaSuccess=0,cudaErrorInvalidValue=1,cudaErrorUnknown=999};
enum cudaMemcpyKind {cudaMemcpyHostToHost=0,cudaMemcpyHostToDevice=1,cudaMemcpyDeviceToHost=2,cudaMemcpyDeviceToDevice=3,cudaMemcpyDefault=4};
namespace strata_test_cuda { inline size_t copies=0; inline bool fail_next_copy=false; }
inline cudaError_t cudaMemcpy(void* dst,const void* src,size_t n,cudaMemcpyKind) {
    ++strata_test_cuda::copies;
    if(strata_test_cuda::fail_next_copy) {strata_test_cuda::fail_next_copy=false;return cudaErrorInvalidValue;}
    if(n && (!dst || !src)) return cudaErrorInvalidValue;
    if(n) std::memcpy(dst,src,n);
    return cudaSuccess;
}
inline cudaError_t cudaDeviceSynchronize() {return cudaSuccess;}
inline constexpr unsigned cudaStreamNonBlocking = 1;
inline cudaError_t cudaGetDevice(int* device) {*device=0;return cudaSuccess;}
inline cudaError_t cudaStreamCreateWithFlags(cudaStream_t* stream,unsigned) {
    static int marker;
    *stream=&marker;
    return cudaSuccess;
}
inline cudaError_t cudaMemcpyAsync(void* dst,const void* src,size_t n,cudaMemcpyKind kind,cudaStream_t) {
    return cudaMemcpy(dst,src,n,kind);
}
inline cudaError_t cudaStreamSynchronize(cudaStream_t) {return cudaSuccess;}
inline cudaError_t cudaGetLastError() {return cudaSuccess;}
inline const char* cudaGetErrorString(cudaError_t e) {return e==cudaSuccess?"success":"test transfer failure";}

// PinnedRing is included by the upstream prefill interface. This host fixture
// must never exercise pinned allocation or event ordering: abort if it does.
inline constexpr unsigned cudaHostAllocPortable = 1;
inline constexpr unsigned cudaEventDisableTiming = 2;
inline cudaError_t cudaHostAlloc(void**, size_t, unsigned) { std::abort(); }
inline cudaError_t cudaFreeHost(void*) { std::abort(); }
inline cudaError_t cudaEventCreateWithFlags(cudaEvent_t*, unsigned) { std::abort(); }
inline cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t) { std::abort(); }
inline cudaError_t cudaEventSynchronize(cudaEvent_t) { std::abort(); }
inline cudaError_t cudaEventDestroy(cudaEvent_t) { std::abort(); }
