#include <iostream>

#include "matrix.hpp"

int main() {
    const auto a = lin_alg::Matrix{ lin_alg::Dimension{ 5, 5 } };
    const auto b = lin_alg::Matrix{ lin_alg::Dimension{ 5, 5 } };
    const auto c = tiled_multiply(a, b, lin_alg::TileSpec{ 1U, 1U, 1U });
    std::cout << display(c.dim()) << std::endl;
}
