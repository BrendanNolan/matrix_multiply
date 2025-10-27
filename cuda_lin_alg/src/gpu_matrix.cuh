#pragma once

#include "matrix.cuh"

namespace cuda_lin_alg {

__global__ void tiled_multiply(const lin_alg::Matrix* A,
        const lin_alg::Matrix* B,
        lin_alg::Matrix* C);

}
