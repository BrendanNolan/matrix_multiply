#include "matrix.hpp"

#include <format>

std::string display(const Dimension& dim) {
    return std::format("({}, {})", dim.i, dim.j);
}

Dimension dimension(const MatrixImpl& matrix) {
    if (matrix.empty())
        return Dimension{ 0U, 0U };
    assert(!matrix.front().empty());
    return Dimension{ matrix.size(), matrix.front().size() };
}

Matrix multiply(const Matrix& a, const Matrix& b, const TileSpec& tile_spec) {
    assert(a.dim() == b.dim());
    auto c = Matrix(Dimension{a.dim().i, b.dim().j});
    return c;
}
