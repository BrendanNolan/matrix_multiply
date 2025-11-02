#pragma once

#include "matrix.hpp"

namespace cuda_lin_alg {

lin_alg::Matrix cuda_tiled_multiply(const lin_alg::Matrix& a,
        const lin_alg::Matrix& b,
        unsigned int tile_size);

}
