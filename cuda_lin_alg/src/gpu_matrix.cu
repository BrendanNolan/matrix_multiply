#include "gpu_matrix.cuh"
#include <cassert>
#include <chrono>
#include <cuda_runtime.h>
#include <iostream>

namespace {

__global__ void tiled_multiply(const float* A,
        const unsigned int ai,
        const unsigned int aj,
        const float* B,
        const unsigned int bj,
        float* C) {
    assert(blockDim.x == blockDim.y);
    const auto T = blockDim.x;
    extern __shared__ float shared[];
    float* a_tile = shared;
    float* b_tile = shared + T * T;
    float* c_tile = shared + 2 * T * T;
    const auto g_i = blockIdx.x * blockDim.x + threadIdx.x;
    const auto g_j = blockIdx.y * blockDim.y + threadIdx.y;
    const auto l_c_cell = threadIdx.x * T + threadIdx.y;
    c_tile[l_c_cell] = 0.0f;
    for (auto k = 0U; k < aj; k += T) {
        const auto in_scope_for_a = (g_i < ai && k + threadIdx.y < aj);
        const auto in_scope_for_b = (k + threadIdx.x < aj && g_j < bj);
        a_tile[l_c_cell] = in_scope_for_a ? A[g_i * aj + (k + threadIdx.y)] : 0U;
        b_tile[l_c_cell] = in_scope_for_b ? B[(k + threadIdx.x) * bj + g_j] : 0U;
        __syncthreads();
        for (auto kk = 0U; kk < T; ++kk) {
            c_tile[l_c_cell] += a_tile[threadIdx.x * T + kk] * b_tile[kk * T + threadIdx.y];
        }
        if (g_i < ai && g_j < bj)
            C[g_i * bj + g_j] = c_tile[l_c_cell];
        __syncthreads();
    }
}

}// namespace

namespace cuda_lin_alg {

lin_alg::Matrix cuda_tiled_multiply(const lin_alg::Matrix& a,
        const lin_alg::Matrix& b,
        unsigned int tile_size) {
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
    const auto grid_dim = dim3{static_cast<unsigned int>((a.dim().i + tile_size - 1) / tile_size),
            static_cast<unsigned int>((b.dim().j + tile_size - 1) / tile_size)};
    const auto shared_mem_size = static_cast<unsigned int>(tile_size * tile_size * 3U);
    const auto start = std::chrono::high_resolution_clock::now();
    tiled_multiply<<<grid_dim, block_dim, shared_mem_size>>>(
            A, a.dim().i, a.dim().j, B, b.dim().j, C);
    cudaDeviceSynchronize();
    const auto end = std::chrono::high_resolution_clock::now();
    const auto duration_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Cuda multiply time: " << duration_ms << " ms" << std::endl;
    float* h_C = static_cast<float*>(malloc(c_bytes * sizeof(float)));
    cudaMemcpy(h_C, C, c_bytes, cudaMemcpyDeviceToHost);
    return lin_alg::Matrix::from_raw(h_C, lin_alg::Dimension{a.dim().i, b.dim().j});
}

}// namespace cuda_lin_alg
