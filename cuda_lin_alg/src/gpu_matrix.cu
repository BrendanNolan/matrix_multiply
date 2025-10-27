#include "gpu_matrix.cuh"

__device__ void copy_elements(const lin_alg::Matrix* source,
        float* target, const size_t);

__global__ void tiled_multiply(const lin_alg::Matrix* A,
        const lin_alg::Matrix* B,
        lin_alg::Matrix* C) {
    const auto T = blockDim.x;
    extern __shared__ float shared[];
    float* a_data = shared;
    float* b_data = shared + T*T;
    float* c_data = shared + 2*T*T;
    const auto tile_dim = lin_alg::Dimension{T, T};
    const auto a = lin_alg::Matrix{a_data, tile_dim};
    const auto b = lin_alg::Matrix{b_data, tile_dim};
    auto c = lin_alg::Matrix{c_data, tile_dim};
    const auto i = blockIdx.x * blockDim.x;
    const auto j = blockIdx.y * blockDim.y;
    // for (auto kk = )
}
