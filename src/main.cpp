#include <iostream>

#include "matrix.hpp"

int main() {
    const auto a = Matrix{Dimension{5, 5}};
    const auto b = Matrix{Dimension{5, 5}};
    const auto c = multiply(a, b, TileSpec{1U, 1U, 1U});
    std::cout << display(c.dim()) << std::endl;
}
