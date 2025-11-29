#pragma once

#include <vector>

namespace cuda_utils {

float* transfer_to_cuda(const float* host_array, const size_t count);
std::vector<float> transfer_from_cuda(const float* device_array, const size_t count);

} // namespace cuda_utils
