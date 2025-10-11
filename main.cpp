#include <cassert>
#include <iostream>
#include <vector>

using Matrix = std::vector<std::vector<float>>;

struct Dimension {
    size_t i = 0U;
    size_t j = 0U;
};

Dimension dimension(const Matrix& matrix) {
    if (matrix.empty())
        return Dimension{ 0U, 0U };
    assert(!matrix.front().empty());
    return Dimension{ matrix.size(), matrix.front().size() };
}

// i is the row count of the LHS tiles
// j is the column count of the RHS tiles
// k is the column count of the LHS tiles (and necessarily the row count of the RHS tiles)
struct TileSpec {
    size_t i = 0U;
    size_t j = 0U;
    size_t k = 0U;
};

Matrix multiply(const Matrix& a, const Matrix& b) {
    assert(dimension(a).j == dimension(b).i);
}

int main() {
}
