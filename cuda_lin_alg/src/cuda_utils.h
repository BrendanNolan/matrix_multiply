#pragma once

extern "C" {
float* allocate_on_cuda(const float* host_array, const size_t count);
float* transfer_from_cuda(const float* device_array, const size_t count);
}
