#include <chrono>
#include <eigen3/Eigen/Dense>

#include "utils.h"

constexpr uint32_t M{4}, K{8}, N{4};

__global__ void matmul(float* a, float* b, float* c, int64_t M, int64_t K, int64_t N)
{
    // TODO: ...
}

RowMatrix gpuMatmul(const RowMatrix& A, const RowMatrix& B)
{
    int64_t M{A.rows()};
    int64_t K{A.cols()};  // Which should match B.rows()
    int64_t N{B.cols()};

    RowMatrix result(M, N);
    result.fill(0.0f);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeof(float) * A.size());
    cudaMalloc(&d_B, sizeof(float) * B.size());
    cudaMalloc(&d_C, sizeof(float) * result.size());
    cudaCheckErrors("malloc failed.");

    cudaMemcpy(d_A, A.data(), sizeof(float) * A.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), sizeof(float) * B.size(), cudaMemcpyHostToDevice);
    cudaCheckErrors("memcpy failed.");

    // TODO: Blocks? Threads?
    // matmul<<<blocks, threads>>>(d_A, d_B, d_C, M, K, N);
    cudaCheckErrors("kernel launch failed");
    cudaDeviceSynchronize();

    cudaMemcpy(result.data(), d_C, sizeof(float) * result.size(), cudaMemcpyDeviceToHost);

    return result;
}

int main()
{
    // NOTE: Eigen matrices are column major order in memory. I've made this alias for
    // a row major matrix.
    RowMatrix A(M, K), B(K, N);

    populateMatrix(A);
    populateMatrix(B);

    RowMatrix cpu_C{A * B};

    RowMatrix gpu_C{gpuMatmul(A, B)};

    checkMatrixApproxEquality(cpu_C, gpu_C, 0.01f);
}
