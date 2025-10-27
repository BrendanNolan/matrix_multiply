#pragma once

#include <cassert>
#include <ostream>
#include <string>
#include <vector>

namespace lin_alg {

struct Dimension {
    size_t i = 0U;
    size_t j = 0U;
    bool operator==(const Dimension& other) const;
    bool operator!=(const Dimension& other) const;
};
std::string display(const Dimension& dim);

class Matrix {
 public:
    Matrix(float* entries, const Dimension& dim);
    ~Matrix();
    static Matrix zeroes(const Dimension& dim);
    static Matrix random(const Dimension& dim);
    Dimension dim() const;
    float operator()(size_t i, size_t j) const;
    float& operator()(size_t i, size_t j);
    bool operator==(const Matrix& other) const;
 private:
    float* impl_ = nullptr;
    Dimension dim_;
};

std::ostream& operator<<(std::ostream& os, const Matrix& matrix);
bool admits_tile(const Matrix& matrix, size_t tile_size);
Matrix naive_multiply(const Matrix& a, const Matrix& b);
Matrix tiled_multiply(const Matrix& a, const Matrix& b, size_t tile_size);

}// namespace lin_alg
