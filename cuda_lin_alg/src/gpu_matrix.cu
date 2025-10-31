#include "gpu_matrix.cuh"
#include <cassert>
#include <cuda_runtime.h>

namespace cuda_lin_alg {

__global__ void tiled_multiply(const float* A,
        const size_t ai,
        const size_t aj,
        const float* B,
        const size_t bj,
        float* C) {
    assert(blockDim.x == blockDim.y);
    const auto T = blockDim.x;
    extern __shared__ float shared[];
    float* a_tile = shared;
    float* b_tile = shared + T * T;
    float* c_tile = shared + 2 * T * T;
    const auto g_corner_i = blockIdx.x * blockDim.x;
    const auto g_corner_j = blockIdx.y * blockDim.y;
    const auto g_i = g_corner_i + threadIdx.x;
    const auto g_j = g_corner_j + threadIdx.y;
    const auto g_c_cell = g_i * aj + g_j;
    if (g_c_cell >= ai * bj)
        return;
    const auto l_c_cell = threadIdx.x * T + threadIdx.y;
    c_tile[l_c_cell] = 0.0f;
    for (auto k = 0U; k < aj; k += T) {
        a_tile[l_c_cell] = A[g_i * aj + k];
        b_tile[l_c_cell] = B[k * bj + g_j];
        __syncthreads();
        for (auto kk = k; kk < min(aj, static_cast<size_t>(k + T)); ++kk) {
            c_tile[l_c_cell] += a_tile[threadIdx.x * T + k] * b_tile[k * T + threadIdx.y];
        }
    }
    __syncthreads();
    C[g_i * bj + g_j] = c_tile[l_c_cell];
    __syncthreads();
}

}// namespace cuda_lin_alg
