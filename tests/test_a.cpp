#include "matrix.hpp"
#include <gtest/gtest.h>

TEST(FirstTest, MultiplyDim) {
    const auto dim = lin_alg::Dimension{10000, 10000};
    const auto a = lin_alg::Matrix::random(dim);
    const auto b = lin_alg::Matrix::random(dim);
    EXPECT_EQ(lin_alg::naive_multiply(a, b), tiled_multiply(a, b, 2));
}
