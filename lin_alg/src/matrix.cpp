#include "matrix.hpp"

#include <random>

namespace lin_alg {

std::string display(const Dimension& dim) {
    return "(" + std::to_string(dim.i) + ", " + std::to_string(dim.j) + ")";
}

bool Dimension::operator==(const Dimension& other) const {
    return this->i == other.i && this->j == other.j;
}

Dimension dimension(const MatrixImpl& matrix) {
    if (matrix.empty())
        return Dimension{ 0U, 0U };
    assert(!matrix.front().empty());
    return Dimension{ matrix.size(), matrix.front().size() };
}

Matrix::Matrix(const MatrixImpl& impl)
    : impl_{ impl } {
}

Matrix::Matrix(const Dimension& dim) {
    const auto row = std::vector<float>(dim.j, 0.0f);
    impl_ = std::vector<std::vector<float>>(dim.i, row);
}

Matrix Matrix::random(const Dimension& dim) {
    std::mt19937 gen(147);
    std::uniform_int_distribution<> dist(0, 100);
    auto matrix = Matrix{ dim };
    for (auto i = 0U; i < dim.i; ++i) {
        for (auto j = 0U; j < dim.j; ++j) {
            matrix[i][j] = dist(gen);
        }
    }
    return matrix;
}

Dimension Matrix::dim() const {
    return dimension(impl_);
}

std::vector<float>& Matrix::operator[](size_t index) {
    return impl_.at(index);
}

const std::vector<float>& Matrix::operator[](size_t index) const {
    return impl_.at(index);
}

bool Matrix::operator==(const Matrix& other) const {
    if (this->dim() != other.dim())
        return false;
    for (auto row = 0U; row < this->dim().i; ++row) {
        if (impl_.at(row) != other.impl_.at(row)) {
            return false;
        }
    }
    return true;
}

MatrixImpl Matrix::raw() const {
    return impl_;
}

std::ostream& operator<<(std::ostream& os, const Matrix& matrix) {
    if (matrix.dim().i == 0U || matrix.dim().j == 0U) {
        os << std::endl;
        return os;
    }
    for (const auto& row : matrix.raw()) {
        for (const auto entry : row) {
            os << entry << ' ';
        }
        os << std::endl;
    }
    return os;
}

bool admits_tile(const Matrix& matrix, size_t tile_size) {
    const auto dim = matrix.dim();
    return tile_size <= dim.i && tile_size <= dim.j;
}

Matrix naive_multiply(const Matrix& a, const Matrix& b) {
    assert(a.dim().j == b.dim().i);
    auto c = Matrix(Dimension{ a.dim().i, b.dim().j });
    return c;
}

Matrix tiled_multiply(const Matrix& a, const Matrix& b, const size_t tile_size) {
    assert(a.dim().j == b.dim().i);
    assert(admits_tile(a, tile_size) && admits_tile(b, tile_size) && tile_size > 0U);
    const auto M = a.dim().i;
    const auto N = b.dim().j;
    const auto K = a.dim().j;
    const auto S = tile_size;
    auto C = Matrix(Dimension{ a.dim().i, b.dim().j });
    for (auto i = 0U; i < M; i += S) {
        for (auto j = 0U; j < N; j += S) {
            // top left of current C block is at (i,j)
            for (auto k = 0U; k < K; k += S) {
                for (auto ii = i; ii < i + S; ++ii) {
                    for (auto jj = j; jj < j + S; ++jj) {
                        for (auto kk = k; kk < k + S; ++kk) {
                            C[ii][jj] += a[ii][kk] * b[kk][jj];
                        }
                    }
                }
            }
        }
    }
    return C;
}

}// namespace lin_alg
