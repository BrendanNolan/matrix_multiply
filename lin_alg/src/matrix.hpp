#pragma once

#include <cassert>
#include <ostream>
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
    Matrix(const MatrixImpl& impl);
    Matrix(const Dimension& dim);
    static Matrix random(const Dimension& dim);
    Dimension dim() const;
    std::vector<float>& operator[](size_t index);
    const std::vector<float>& operator[](size_t index) const;
    bool operator==(const Matrix& other) const;
    MatrixImpl raw() const;
 private:
    MatrixImpl impl_;
};

std::ostream& operator<<(std::ostream& os, const Matrix& matrix);
bool admits_tile(const Matrix& matrix, size_t tile_size);
Matrix naive_multiply(const Matrix& a, const Matrix& b);
Matrix tiled_multiply(const Matrix& a, const Matrix& b, size_t tile_size);

}// namespace lin_alg
