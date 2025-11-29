#include "cuda_utils.hpp"

#include <cuda_runtime.h>

namespace cuda_utils {

float* transfer_to_cuda(const float* host_array, const size_t count) {
    float* device_memory;
    cudaMalloc(&device_memory, count);
    cudaMemcpy(device_memory, host_array, count, cudaMemcpyHostToDevice);
}

std::vector<float> transfer_from_cuda(const float* device_array, const size_t count) {
    auto host_memory = std::vector<float>{};
    host_memory.resize(count);
    cudaMemcpy(host_memory.data(), device_array, count, cudaMemcpyDeviceToHost);
    return host_memory;
}

} // namespace cuda_utils
