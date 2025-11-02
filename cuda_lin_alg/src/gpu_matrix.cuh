#pragma once

#include <cuda_runtime.h>

namespace cuda_lin_alg {

__global__ void tiled_multiply(const float* A,
        const unsigned int ai,
        const unsigned int aj,
        const float* B,
        const unsigned int bj,
        float* C);

}// namespace cuda_lin_alg
