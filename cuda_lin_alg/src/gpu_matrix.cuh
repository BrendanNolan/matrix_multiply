#pragma once

namespace cuda_lin_alg {

__global__ void tiled_multiply(const float* A,
        const size_t ai,
        const size_t aj,
        const float* B,
        const size_t bj,
        float* C);

}
