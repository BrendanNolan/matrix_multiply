#include "gpu_matrix.cuh"
#include <cassert>
#include <cuda_runtime.h>

namespace {

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

}// namespace

namespace cuda_lin_alg {

lin_alg::Matrix
        cuda_tiled_multiply(const lin_alg::Matrix& a, const lin_alg::Matrix& b, size_t tile_size) {
    const auto a_bytes = raw_size(a) * sizeof(float);
    float* A;
    cudaMalloc(&A, a_bytes);
    cudaMemcpy(A, a.raw(), a_bytes, cudaMemcpyHostToDevice);
    const auto b_bytes = raw_size(b) * sizeof(float);
    float* B;
    cudaMalloc(&B, b_bytes);
    cudaMemcpy(B, b.raw(), b_bytes, cudaMemcpyHostToDevice);
    const auto c_bytes = a.dim().i * b.dim().j * sizeof(float);
    float* C;
    cudaMalloc(&C, c_bytes);
    const auto block_dim =
            dim3{static_cast<unsigned int>(tile_size), static_cast<unsigned int>(tile_size)};
    const auto grid_dim = dim3{static_cast<unsigned int>(a.dim().i + 1 / tile_size),
            static_cast<unsigned int>(b.dim().j + 1 / tile_size)};
    tiled_multiply<<<grid_dim, block_dim, 1 << 12>>>(A, a.dim().i, a.dim().j, B, b.dim().j, C);
    float* h_C = static_cast<float*>(malloc(c_bytes * sizeof(float)));
    cudaMemcpy(C, h_C, c_bytes, cudaMemcpyDeviceToHost);
    return lin_alg::Matrix{h_C, lin_alg::Dimension{a.dim().i, b.dim().j}};
}

}// namespace cuda_lin_alg
