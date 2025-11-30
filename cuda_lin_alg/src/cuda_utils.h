#pragma once

extern "C" {
float* transfer_to_cuda(const float* host_array, const size_t count);
float* transfer_from_cuda(const float* device_array, const size_t count);
}
