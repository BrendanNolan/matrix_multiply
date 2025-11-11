#include "gpu_matrix.cuh"
#include "matrix.hpp"
#include "test_config.hpp"

#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

#include <gtest/gtest.h>

namespace {

struct LaunchConfig {
    dim3 grid_dim;
    dim3 block_dim;
};
unsigned int tile_size(const LaunchConfig& config) {
    assert(config.block_dim.x == config.block_dim.y);
    return config.block_dim.x * config.block_dim.x;
}
unsigned int shared_mem_size(const LaunchConfig& config) {
    return tile_size(config) * 3U * sizeof(float);
}

std::string to_string(const dim3& dim) {
    return "(" + std::to_string(dim.x) + "," + std::to_string(dim.y) + "," + std::to_string(dim.z)
            + ")";
}

std::string to_string(const LaunchConfig& config) {
    return "gridDim: " + to_string(config.grid_dim) + " blockDim: " + to_string(config.block_dim);
}

using Dim = lin_alg::Dimension;

struct CudaInput {
    const float* A = nullptr;
    const unsigned int ai = 0U;
    const unsigned int aj = 0U;
    const float* B = nullptr;
    const unsigned int bj = 0U;
    float* C = nullptr;
    LaunchConfig config;
};

std::chrono::milliseconds raw_cuda_multiply(const CudaInput& input) {
    std::cout << "shared_mem_size: " << shared_mem_size(input.config) << std::endl;
    const auto start = std::chrono::high_resolution_clock::now();
    cuda_lin_alg::tiled_multiply<<<input.config.grid_dim,
            input.config.block_dim,
            shared_mem_size(input.config)>>>(
            input.A, input.ai, input.aj, input.B, input.bj, input.C);
    cudaDeviceSynchronize();
    const auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
}

CudaInput ExtractInput(const lin_alg::Matrix& a,
        const lin_alg::Matrix& b,
        const std::optional<LaunchConfig>& optional_config) {
    const auto a_bytes = raw_size(a) * sizeof(float);
    float* A;
    cudaMalloc(&A, a_bytes);
    cudaMemcpy(A, a.raw(), a_bytes, cudaMemcpyHostToDevice);
    const auto b_bytes = raw_size(b) * sizeof(float);
    float* B;
    cudaMalloc(&B, b_bytes);
    cudaMemcpy(B, b.raw(), b_bytes, cudaMemcpyHostToDevice);
    const auto c_bytes = a.dim().i * b.dim().j * sizeof(float);
    float* C;
    cudaMalloc(&C, c_bytes);
    const auto default_block_edge_size = 4U;
    const auto default_launch_config = LaunchConfig{
            .grid_dim = dim3{static_cast<unsigned int>((a.dim().i + default_block_edge_size - 1)
                                     / default_block_edge_size),
                    static_cast<unsigned int>(
                            (b.dim().j + default_block_edge_size - 1) / default_block_edge_size)},
            .block_dim = dim3{default_block_edge_size, default_block_edge_size}};
    return CudaInput{.A = A,
            .ai = a.dim().i,
            .aj = a.dim().j,
            .B = B,
            .bj = b.dim().j,
            .C = C,
            .config = optional_config.value_or(default_launch_config)};
}

struct MultiplyResult {
    lin_alg::Matrix result_matrix;
    std::chrono::milliseconds duration;
    LaunchConfig launch_config_used;
};
std::string to_string(const MultiplyResult& result) {
    return "duration: " + std::to_string(result.duration.count())
            + "ms, launch config :" + to_string(result.launch_config_used);
}

MultiplyResult cuda_tiled_multiply(const lin_alg::Matrix& a,
        const lin_alg::Matrix& b,
        const std::optional<LaunchConfig>& optional_config = std::nullopt) {
    const auto input = ExtractInput(a, b, optional_config);
    const auto duration_ms = raw_cuda_multiply(input);
    const auto c_bytes = input.ai * input.bj * sizeof(float);
    float* h_C = static_cast<float*>(malloc(c_bytes));
    cudaMemcpy(h_C, input.C, c_bytes, cudaMemcpyDeviceToHost);
    return MultiplyResult{.result_matrix = lin_alg::Matrix::from_raw(
                                  h_C, lin_alg::Dimension{a.dim().i, b.dim().j}),
            .duration = duration_ms,
            .launch_config_used = input.config};
}

std::vector<LaunchConfig> generate_launch_configs(int dim_of_square_matrix) {
    auto configs = std::vector<LaunchConfig>{};
    const auto tile_sizes = std::vector<unsigned int>{1U, 8U, 16U, 32U};
    for (const auto tile_size : tile_sizes) {
        const auto blocks = dim_of_square_matrix / tile_size;
        const auto grid = dim3(blocks, blocks);
        const auto block = dim3(tile_size, tile_size);
        configs.push_back({grid, block});
    }
    return configs;
}

void correctness_test(const unsigned int rows_left,
        const unsigned int common,
        const unsigned int columns_right) {
    const auto a = lin_alg::Matrix::random(Dim{rows_left, common});
    const auto b = lin_alg::Matrix::random(Dim{common, columns_right});
    const auto naive_multiply_result = lin_alg::naive_multiply(a, b);
    const auto cuda_multiply_result = cuda_tiled_multiply(a, b);
    std::cout << "Used config " << to_string(cuda_multiply_result.launch_config_used) << std::endl;
    EXPECT_EQ(cuda_multiply_result.result_matrix, naive_multiply_result);
    for (const auto& config : generate_launch_configs(common)) {
        std::cout << "Using config: " << to_string(config) << std::endl;
        const auto cuda_multiply_result = cuda_tiled_multiply(a, b);
        EXPECT_EQ(cuda_multiply_result.result_matrix, naive_multiply_result);
    }
}

void comparative_speed_test(const unsigned int dim_of_square_matrix, const LaunchConfig& config) {
    const auto a = lin_alg::Matrix::random(Dim{dim_of_square_matrix, dim_of_square_matrix});
    const auto b = lin_alg::Matrix::random(Dim{dim_of_square_matrix, dim_of_square_matrix});

    auto start = std::chrono::high_resolution_clock::now();
    const auto naive_multiply_result = lin_alg::naive_multiply(a, b);
    auto end = std::chrono::high_resolution_clock::now();
    const auto naive_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Naive CPU execution time: " << naive_time << " ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    auto tiled_multiply_result = lin_alg::tiled_multiply(a, b, 4U);
    end = std::chrono::high_resolution_clock::now();
    const auto optimised_cpu_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Optimised CPU execution time: " << optimised_cpu_time << " ms" << std::endl;

    const auto cuda_multiply_result = cuda_tiled_multiply(a, b, config);
    std::cout << "Optimised GPU execution " << to_string(cuda_multiply_result) << std::endl;

    EXPECT_EQ(tiled_multiply_result, naive_multiply_result);
    EXPECT_EQ(cuda_multiply_result.result_matrix, naive_multiply_result);
}

void many_config_speed_test(const unsigned int dim_of_square_matrix) {
    const auto a = lin_alg::Matrix::random(Dim{dim_of_square_matrix, dim_of_square_matrix});
    const auto b = lin_alg::Matrix::random(Dim{dim_of_square_matrix, dim_of_square_matrix});
    for (const auto& config : generate_launch_configs(dim_of_square_matrix)) {
        const auto input = ExtractInput(a, b, config);
        const auto duration = raw_cuda_multiply(input);
        std::cout << "With " << dim_of_square_matrix << "-square matrices and config "
                  << to_string(config) << ", CUDA execution took " << duration.count() << " ms"
                  << std::endl;
    }
}

LaunchConfig get_test_config() {
    return LaunchConfig{
            .grid_dim =
                    dim3{TestConfig::instance().block_edge, TestConfig::instance().block_edge, 1U},
            .block_dim =
                    dim3{TestConfig::instance().block_edge, TestConfig::instance().block_edge, 1U}};
}

}// namespace

TEST(SpeedTest, SevenElements) {
    comparative_speed_test(7U, get_test_config());
}

TEST(SpeedTest, ThirtyThreeElements) {
    comparative_speed_test(33U, get_test_config());
}

TEST(SpeedTest, OneMillionElements) {
    const auto square_dim = 1U << 10U;
    comparative_speed_test(square_dim, get_test_config());
    for (const auto& config : generate_launch_configs(square_dim)) {
        comparative_speed_test(square_dim, config);
    }
}

TEST(SpeedTest, OneThousandElements) {
    comparative_speed_test(1U << 7U, get_test_config());
}

TEST(ManyConfigSpeedTest, OneMillionElements) {
    many_config_speed_test(1U << 10U);
}

TEST(ManyConfigSpeedTest, OneThousandElements) {
    many_config_speed_test(1U << 7U);
}

TEST(CorrectnessTest, Small) {
    const auto rows_left = (1U << 5) + 1U;
    const auto common = (1U << 4) + 3U;
    const auto columns_right = (1U << 6) + 1U;
    correctness_test(rows_left, common, columns_right);
}

TEST(CorrectnessTest, Large) {
    const auto rows_left = (1U << 8) + 1U;
    const auto common = (1U << 7) + 3U;
    const auto columns_right = (1U << 6) + 1U;
    correctness_test(rows_left, common, columns_right);
}
