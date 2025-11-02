#include <cassert>

namespace cuda_lin_alg {

__global__ void tiled_multiply(const float* A,
        const unsigned int ai,
        const unsigned int aj,
        const float* B,
        const unsigned int bj,
        float* C) {
    assert(blockDim.x == blockDim.y);
    const auto T = blockDim.x;
    extern __shared__ float shared[];
    float* a_tile = shared;
    float* b_tile = shared + T * T;
    float* c_tile = shared + 2 * T * T;
    const auto g_i = blockIdx.x * blockDim.x + threadIdx.x;
    const auto g_j = blockIdx.y * blockDim.y + threadIdx.y;
    const auto l_c_cell = threadIdx.x * T + threadIdx.y;
    c_tile[l_c_cell] = 0.0f;
    for (auto k = 0U; k < aj; k += T) {
        const auto in_scope_for_a = (g_i < ai && k + threadIdx.y < aj);
        const auto in_scope_for_b = (k + threadIdx.x < aj && g_j < bj);
        a_tile[l_c_cell] = in_scope_for_a ? A[g_i * aj + (k + threadIdx.y)] : 0U;
        b_tile[l_c_cell] = in_scope_for_b ? B[(k + threadIdx.x) * bj + g_j] : 0U;
        __syncthreads();
        for (auto kk = 0U; kk < T; ++kk) {
            c_tile[l_c_cell] += a_tile[threadIdx.x * T + kk] * b_tile[kk * T + threadIdx.y];
        }
        __syncthreads();
    }
    if (g_i < ai && g_j < bj)
        C[g_i * bj + g_j] = c_tile[l_c_cell];
}

}// namespace cuda_lin_alg
