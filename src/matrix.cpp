#include "matrix.hpp"

namespace lin_alg {

std::string display(const Dimension& dim) {
    return "(" + std::to_string(dim.i) + ", " + std::to_string(dim.j) + ")";
}

Dimension dimension(const MatrixImpl& matrix) {
    if (matrix.empty())
        return Dimension{ 0U, 0U };
    assert(!matrix.front().empty());
    return Dimension{ matrix.size(), matrix.front().size() };
}

Matrix naive_multiply(const Matrix& a, const Matrix& b) {
    assert(a.dim() == b.dim());
    auto c = Matrix(Dimension{ a.dim().i, b.dim().j });
    return c;
}

Matrix tiled_multiply(const Matrix& a, const Matrix& b, const TileSpec& tile_spec) {
    assert(a.dim() == b.dim());
    auto c = Matrix(Dimension{ a.dim().i, b.dim().j });
    return c;
}

}// namespace lin_alg
