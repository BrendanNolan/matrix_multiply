#include "matrix.hpp"
#include <gtest/gtest.h>

TEST(FirstTest, MultiplyDim) {
    const auto a = lin_alg::Matrix::random(lin_alg::Dimension{ 8, 8 });
    const auto b = lin_alg::Matrix::random(lin_alg::Dimension{ 8, 8 });
    EXPECT_TRUE(lin_alg::naive_multiply(a, b) == tiled_multiply(a, b, 2));
}
