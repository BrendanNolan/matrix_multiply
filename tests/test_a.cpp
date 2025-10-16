#include "matrix.hpp"
#include <gtest/gtest.h>

TEST(FirstTest, MultiplyDim) {
    const auto a = lin_alg::Matrix{ lin_alg::Dimension{ 5, 5 } };
    const auto b = lin_alg::Matrix{ lin_alg::Dimension{ 5, 5 } };
    const auto c = tiled_multiply(a, b, 1);
    const auto expected_dimension = lin_alg::Dimension{ 5, 5 };
    EXPECT_TRUE(c.dim() == expected_dimension);
}
