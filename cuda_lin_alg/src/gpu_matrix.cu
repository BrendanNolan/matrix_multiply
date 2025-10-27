#include "gpu_matrix.cuh"

__device__ void copy_elements(const float* source,
        const size_t start,
        const size_t count,
        float* target);

__global__ void tiled_multiply(const float* A,
        const float* B,
        float* C) {
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
