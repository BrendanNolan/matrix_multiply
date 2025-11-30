#include "cuda_utils.h"

#include <cuda_runtime.h>

double* transfer_to_cuda(const double* host_array, const size_t count) {
    double* device_memory;
    cudaMalloc(&device_memory, count);
    cudaMemcpy(device_memory, host_array, count, cudaMemcpyHostToDevice);
    return device_memory;
}

double* transfer_from_cuda(const double* device_array, const size_t count) {
    auto* host_memory = static_cast<double*>(malloc(sizeof(double) * count));
    cudaMemcpy(host_memory, device_array, count, cudaMemcpyDeviceToHost);
    return host_memory;
}
