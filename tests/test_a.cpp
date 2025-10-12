#include <gtest/gtest.h>
#include "matrix.hpp"

TEST(FirstTest, MultiplyDim) {
    const auto a = lin_alg::Matrix{ lin_alg::Dimension{ 5, 5 } };
    const auto b = lin_alg::Matrix{ lin_alg::Dimension{ 5, 5 } };
    const auto c = tiled_multiply(a, b, lin_alg::TileSpec{ 1U, 1U, 1U });
    EXPECT_TRUE(c.dim(), lin_alg::Dimension{5, 5});
}
