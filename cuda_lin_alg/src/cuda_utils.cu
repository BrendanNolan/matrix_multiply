#include "cuda_utils.h"

#include <cuda_runtime.h>

float* transfer_to_cuda(const float* host_array, const size_t count) {
    float* device_memory;
    cudaMalloc(&device_memory, count);
    cudaMemcpy(device_memory, host_array, count, cudaMemcpyHostToDevice);
    return device_memory;
}

float* transfer_from_cuda(const float* device_array, const size_t count) {
    auto* host_memory = static_cast<float*>(malloc(sizeof(float) * count));
    cudaMemcpy(host_memory, device_array, count, cudaMemcpyDeviceToHost);
    return host_memory;
}
