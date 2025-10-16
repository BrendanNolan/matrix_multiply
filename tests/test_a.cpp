#include "matrix.hpp"

#include <chrono>
#include <iostream>

#include <gtest/gtest.h>

TEST(FirstTest, MultiplyDim) {
    const auto dim = lin_alg::Dimension{10000, 10000};
    const auto a = lin_alg::Matrix::random(dim);
    const auto b = lin_alg::Matrix::random(dim);
    const auto tiled_multiply_result = lin_alg::tiled_multiply(a, b, 4U);
    const auto naive_multiply_result = lin_alg::naive_multiply(a, b);
    EXPECT_EQ(tiled_multiply_result, naive_multiply_result);
}
