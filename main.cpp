#include <cassert>
#include <iostream>
#include <vector>

using MatrixImpl = std::vector<std::vector<float>>;

struct Dimension {
    size_t i = 0U;
    size_t j = 0U;
    bool operator==(const Dimension& other) const = default;
};

Dimension dimension(const MatrixImpl& matrix) {
    if (matrix.empty())
        return Dimension{ 0U, 0U };
    assert(!matrix.front().empty());
    return Dimension{ matrix.size(), matrix.front().size() };
}

class Matrix {
public:
    Matrix(const MatrixImpl& impl)
        : impl_{impl} {}
    Matrix(const Dimension& dim) {
        const auto row = std::vector<float>(dim.j, 0.0f);
        impl_ = std::vector<std::vector<float>>(dim.i, row);
    }
    Dimension dim() const {
        return dimension(impl_);
    }
private:
    MatrixImpl impl_;
};

// i is the row count of the LHS tiles
// j is the column count of the RHS tiles
// k is the column count of the LHS tiles (and necessarily the row count of the RHS tiles)
struct TileSpec {
    size_t i = 0U;
    size_t j = 0U;
    size_t k = 0U;
};

Matrix multiply(const Matrix& a, const Matrix& b, const TileSpec& tile_spec) {
    assert(a.dim() == b.dim());
    auto c = Matrix(Dimension{a.dim().i, b.dim().j});
    return c;
}

int main() {
    const auto a = Matrix{Dimension{5, 5}};
    const auto b = Matrix{Dimension{5, 5}};
    const auto c = multiply(a, b, TileSpec{1U, 1U, 1U});
    std::cout << c.dim().i << "," << c.dim().j << std::endl;
}
