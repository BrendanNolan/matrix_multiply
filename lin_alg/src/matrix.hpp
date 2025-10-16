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

Matrix naive_multiply(const Matrix& a, const Matrix& b);
Matrix tiled_multiply(const Matrix& a, const Matrix& b, size_t tile_size);

}// namespace lin_alg
