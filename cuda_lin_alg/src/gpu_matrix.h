#pragma once

#include <cuda_runtime.h>

extern "C" {
void launch_tiled_multiply(const double* A,
        const unsigned int ai,
        const unsigned int aj,
        const double* B,
        const unsigned int bj,
        double* C,
        const dim3& grid,
        const dim3& block,
        const unsigned int shared_mem_size);
}

__global__ void tiled_multiply(const double* A,
        const unsigned int ai,
        const unsigned int aj,
        const double* B,
        const unsigned int bj,
        double* C);
