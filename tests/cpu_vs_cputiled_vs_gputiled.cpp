#include "gpu_matrix.cuh"
#include "matrix.hpp"

#include <chrono>
#include <cmath>
#include <iostream>

#include <gtest/gtest.h>

using Dim = lin_alg::Dimension;

TEST(FirstTest, MultiplyDim) {
    const auto a = lin_alg::Matrix::random(Dim{1 << 11, 1 << 10});
    const auto b = lin_alg::Matrix::random(Dim{1 << 10, 1 << 12});

    auto start = std::chrono::high_resolution_clock::now();
    const auto naive_multiply_result = lin_alg::naive_multiply(a, b);
    auto end = std::chrono::high_resolution_clock::now();
    const auto naive_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Naive multiply time: " << naive_time << " ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    auto tiled_multiply_result = lin_alg::tiled_multiply(a, b, 4U);
    end = std::chrono::high_resolution_clock::now();
    const auto tiled_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Tiled multiply time: " << tiled_time << " ms" << std::endl;

    // This function will print its own timing
    const auto cuda_multiply_result = cuda_lin_alg::cuda_tiled_multiply(a, b, 4U);

    EXPECT_EQ(tiled_multiply_result, naive_multiply_result);
    EXPECT_EQ(cuda_multiply_result, naive_multiply_result);
}
