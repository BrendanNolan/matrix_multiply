#include "gpu_matrix.cuh"
#include "matrix.hpp"

#include <chrono>
#include <cmath>
#include <iostream>

#include <gtest/gtest.h>

namespace {

using Dim = lin_alg::Dimension;

lin_alg::Matrix cuda_tiled_multiply(const lin_alg::Matrix& a,
        const lin_alg::Matrix& b,
        const unsigned int tile_size,
        const bool print_timing) {
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
    cuda_lin_alg::tiled_multiply<<<grid_dim, block_dim, shared_mem_size>>>(
            A, a.dim().i, a.dim().j, B, b.dim().j, C);
    cudaDeviceSynchronize();
    const auto end = std::chrono::high_resolution_clock::now();
    const auto duration_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    if (print_timing)
        std::cout << "Optimised GPU execution time: " << duration_ms << " ms" << std::endl;
    float* h_C = static_cast<float*>(malloc(c_bytes * sizeof(float)));
    cudaMemcpy(h_C, C, c_bytes, cudaMemcpyDeviceToHost);
    return lin_alg::Matrix::from_raw(h_C, lin_alg::Dimension{a.dim().i, b.dim().j});
}

void correctness_test(const unsigned int rows_left,
        const unsigned int common,
        const unsigned int columns_right) {
    const auto a = lin_alg::Matrix::random(Dim{rows_left, common});
    const auto b = lin_alg::Matrix::random(Dim{common, columns_right});
    const auto naive_multiply_result = lin_alg::naive_multiply(a, b);
    const auto cuda_multiply_result = cuda_tiled_multiply(a, b, 4U, false);
    EXPECT_EQ(cuda_multiply_result, naive_multiply_result);
}

void speed_test(const unsigned int dim_of_square_matrix) {
    const auto a = lin_alg::Matrix::random(Dim{dim_of_square_matrix, dim_of_square_matrix});
    const auto b = lin_alg::Matrix::random(Dim{dim_of_square_matrix, dim_of_square_matrix});

    auto start = std::chrono::high_resolution_clock::now();
    const auto naive_multiply_result = lin_alg::naive_multiply(a, b);
    auto end = std::chrono::high_resolution_clock::now();
    const auto naive_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Naive CPU execution time: " << naive_time << " ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    auto tiled_multiply_result = lin_alg::tiled_multiply(a, b, 4U);
    end = std::chrono::high_resolution_clock::now();
    const auto tiled_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Optimised CPU execution time: " << tiled_time << " ms" << std::endl;

    const auto cuda_multiply_result = cuda_tiled_multiply(a, b, 4U, true);

    EXPECT_EQ(tiled_multiply_result, naive_multiply_result);
    EXPECT_EQ(cuda_multiply_result, naive_multiply_result);
}

}// namespace

TEST(SpeedTest, OneMillionElements) {
    speed_test(1U << 10U);
}

TEST(SpeedTest, OneThousandElements) {
    speed_test(1U << 7U);
}

TEST(CorrectnessTest, Small) {
    const auto rows_left = (1U << 5) + 1U;
    const auto common = (1U << 4) + 3U;
    const auto columns_right = (1U << 6) + 1U;
    correctness_test(rows_left, common, columns_right);
}

TEST(CorrectnessTest, Large) {
    const auto rows_left = (1U << 8) + 1U;
    const auto common = (1U << 7) + 3U;
    const auto columns_right = (1U << 6) + 1U;
    correctness_test(rows_left, common, columns_right);
}
