#pragma once

extern "C" {
double* transfer_to_cuda(const double* host_array, const size_t count);
double* transfer_from_cuda(const double* device_array, const size_t count);
}
