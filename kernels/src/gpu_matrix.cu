#include "gpu_matrix.cuh"

__host__ __device__ size_t total(const Pos2& pos) {
    return pos.i * pos.j;
}

__global__ void tiled_multiply(const float* A,
        const Pos2& size_A,
        const float* B,
        const Pos2& size_B,
        float* C) {
}
