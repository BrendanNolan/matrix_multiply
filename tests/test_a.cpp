#include "gpu_matrix.cuh"
#include "matrix.hpp"

#include <chrono>
#include <cmath>
#include <iostream>

#include <gtest/gtest.h>

using Dim = lin_alg::Dimension;

TEST(FirstTest, MultiplyDim) {
    auto pow2 = [](size_t x) {
        auto result = 1U;
        while (x != 0U) {
            result *= 2U;
            --x;
        }
        return result;
    };
    const auto a = lin_alg::Matrix::random(Dim{pow2(11U), pow2(10U)});
    const auto b = lin_alg::Matrix::random(Dim{pow2(10U), pow2(12U)});

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

    start = std::chrono::high_resolution_clock::now();
    const auto cuda_multiply_result = cuda_lin_alg::cuda_tiled_multiply(a, b, 4U);
    end = std::chrono::high_resolution_clock::now();
    const auto cuda_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "cuda multiply time: " << cuda_time << " ms" << std::endl;

    EXPECT_EQ(tiled_multiply_result, naive_multiply_result);
    EXPECT_EQ(cuda_multiply_result, naive_multiply_result);
}
