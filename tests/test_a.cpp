#include "matrix.hpp"

#include <chrono>
#include <iostream>

#include <gtest/gtest.h>

TEST(FirstTest, MultiplyDim) {
    const auto dim = lin_alg::Dimension{ 2048U, 2048U };
    const auto a = lin_alg::Matrix::random(dim);
    const auto b = lin_alg::Matrix::random(dim);
    auto start = std::chrono::high_resolution_clock::now();
    auto tiled_multiply_result = lin_alg::tiled_multiply(a, b, 4U);
    auto end = std::chrono::high_resolution_clock::now();
    auto tiled_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Tiled multiply time: " << tiled_time << " ms" << std::endl;
    start = std::chrono::high_resolution_clock::now();
    const auto naive_multiply_result = lin_alg::naive_multiply(a, b);
    end = std::chrono::high_resolution_clock::now();
    auto naive_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Naive multiply time: " << naive_time << " ms" << std::endl;
    std::cout << "Tiled speedup " << static_cast<float>(naive_time) / static_cast<float>(tiled_time)
              << std::endl;
    EXPECT_EQ(tiled_multiply_result, naive_multiply_result);
}
