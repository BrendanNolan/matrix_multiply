#pragma once

#include <cassert>
#include <string>
#include <vector>

namespace lin_alg {

using MatrixImpl = std::vector<std::vector<float>>;

struct Dimension {
    size_t i = 0U;
    size_t j = 0U;
    bool operator==(const Dimension& other) const;
};
std::string display(const Dimension& dim);
Dimension dimension(const MatrixImpl& matrix);

class Matrix {
 public:
    Matrix(const MatrixImpl& impl)
        : impl_{ impl } {
    }
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

Matrix naive_multiply(const Matrix& a, const Matrix& b);
Matrix tiled_multiply(const Matrix& a, const Matrix& b, const TileSpec& tile_spec);

}// namespace lin_alg
