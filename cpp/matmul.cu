#include <chrono>
#include <eigen3/Eigen/Dense>
#include <iostream>

#include "cublas_v2.h"
#include "utils.h"

enum class MatMulImpl
{
    Naive,
    SharedMem,
    cuBLAS
};

#define CUBLAS_CHECK(call)                                                  \
    do                                                                      \
    {                                                                       \
        cublasStatus_t status = call;                                       \
        if (status != CUBLAS_STATUS_SUCCESS)                                \
        {                                                                   \
            fprintf(stderr, "cuBLAS error at %s:%d\n", __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

__global__ void matmul(float* a, float* b, float* c, int64_t m, int64_t k, int64_t n)
{
    // Matrix multiplication between a and b, storing in c.
    // a is M x K.
    // b is K x N.
    // c is M X N;

    // First iterate over every element in c.
    int64_t col{blockIdx.x * blockDim.x + threadIdx.x};
    int64_t row{blockIdx.y * blockDim.y + threadIdx.y};

    if (row >= m || col >= n)
    {
        return;
    }

    c[row * n + col] = 0.0f;
    for (std::size_t i{0}; i < k; ++i)
    {
        c[row * n + col] += a[row * k + i] * b[col + i * n];
    }
}

__global__ void matmul_shm(float* a, float* b, float* c, int64_t m, int64_t k, int64_t n)
{
    // Matrix multiplication between a and b, storing in c.
    // a is M x K.
    // b is K x N.
    // c is M X N;

    // TODO: Dynamically allocate shared memory.
    // TODO: Remove assumption of hardcoded block size.
    __shared__ float shm_a[32][32];
    __shared__ float shm_b[32][32];

    int64_t col{blockIdx.x * blockDim.x + threadIdx.x};
    int64_t row{blockIdx.y * blockDim.y + threadIdx.y};

    // TODO: Assuming block size is 32.
    std::size_t blockSize{32};

    float element{0.0f};
    for (std::size_t i{0}; i < (k + blockSize - 1) / blockSize; ++i)
    {
        std::size_t colOffset{i * blockSize + threadIdx.x};
        std::size_t indexA{colOffset + k * row};
        if (row < m && colOffset < k)
        {
            shm_a[threadIdx.y][threadIdx.x] = a[indexA];
        }
        else
        {
            shm_a[threadIdx.y][threadIdx.x] = 0.0f;
        }

        std::size_t rowOffset{i * blockSize + threadIdx.y};
        std::size_t indexB{rowOffset * n + col};
        if (col < n && rowOffset < k)
        {
            shm_b[threadIdx.y][threadIdx.x] = b[indexB];
        }
        else
        {
            shm_b[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Step 2, compute row x column dot product from shared memory.
        for (std::size_t j{0}; j < blockSize; ++j)
        {
            element += shm_a[threadIdx.y][j] * shm_b[j][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < m && col < n)
    {
        c[row * n + col] = element;
    }
}

void populateMatrix(RowMatrix& mat)
{
    for (std::size_t i{0}; i < mat.rows(); ++i)
    {
        for (std::size_t j{0}; j < mat.cols(); ++j)
        {
            mat(i, j) = (rand() % 1000 - 500) / 500.0;
        }
    }
}

RowMatrix matmulCuda(const RowMatrix& matA, const RowMatrix& matB, MatMulImpl impl)
{
    RowMatrix result(matA.rows(), matB.cols());
    const int64_t M{matA.rows()};
    const int64_t K{matA.cols()};
    const int64_t N{matB.cols()};

    auto tMem = std::chrono::high_resolution_clock::now();
    float *gpuMatA, *gpuMatB, *gpuMatResult;
    cudaMalloc((void**)&gpuMatA, sizeof(float) * matA.size());
    cudaMalloc((void**)&gpuMatB, sizeof(float) * matB.size());
    cudaMalloc((void**)&gpuMatResult, sizeof(float) * result.size());

    cudaMemcpy(gpuMatA, matA.data(), sizeof(float) * matA.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuMatB, matB.data(), sizeof(float) * matB.size(), cudaMemcpyHostToDevice);

    dim3 blockSize(32, 32);
    dim3 blocks{static_cast<uint32_t>((N + blockSize.y - 1) / blockSize.y),
                static_cast<uint32_t>((M + blockSize.x - 1) / blockSize.x)};

    auto tKernel{std::chrono::high_resolution_clock::now()};
    switch (impl)
    {
        case MatMulImpl::Naive:
        {
            std::cout << "Using naive matmul.\n";
            matmul<<<blocks, blockSize>>>(gpuMatA, gpuMatB, gpuMatResult, matA.rows(), matA.cols(), matB.cols());
            break;
        }
        case MatMulImpl::SharedMem:
        {
            std::cout << "Using shared memory matmul.\n";
            matmul_shm<<<blocks, blockSize>>>(gpuMatA, gpuMatB, gpuMatResult, matA.rows(), matA.cols(), matB.cols());
            break;
        }
        case MatMulImpl::cuBLAS:
        {
            cublasHandle_t handle;
            CUBLAS_CHECK(cublasCreate(&handle));
            const float alpha = 1.0f;
            const float beta = 0.0f;
            std::cout << "Calling cublasSgemm" << std::endl;
            CUBLAS_CHECK(cublasSgemm(handle,
                                     CUBLAS_OP_N,  // transA for cuBLAS (B^T)
                                     CUBLAS_OP_N,  // transB for cuBLAS (A^T)
                                     N,            // m (rows of C in cublas, which is cols of B in row-major)
                                     M,            // n (cols of C in cublas, which is rows of A in row-major)
                                     K,            // k (common dimension)
                                     &alpha,
                                     gpuMatB,  // d_A (input to cublas is B)
                                     N,  // lda (leading dim of B, which is N for row-major B if seen as col-major)
                                     gpuMatA,  // d_B (input to cublas is A)
                                     K,  // ldb (leading dim of A, which is K for row-major A if seen as col-major)
                                     &beta,
                                     gpuMatResult,  // d_C (output)
                                     N));  // ldc (leading dim of C, which is N for row-major C if seen as col-major)

            break;
        }

        default:
            std::cerr << "Invalid operation!\n";
            exit(1);

            cudaDeviceSynchronize();
            std::cout << "Kernel computation time: ";
            reportTime(tKernel);
    }
    cudaDeviceSynchronize();
    std::cout << "Kernel computation time: ";
    reportTime(tKernel);

    cudaMemcpy(result.data(), gpuMatResult, sizeof(float) * result.size(), cudaMemcpyDeviceToHost);

    cudaFree(gpuMatA);
    cudaFree(gpuMatB);
    cudaFree(gpuMatResult);

    return result;
}

int main()
{
    // 1000 x 1000 x 1000 takes about ~10-11 seconds in CPU.
    // 20000 x 10000 x 20000 takes about 7 seconds with Naive, ~1-2s with SharedMem.
    constexpr std::size_t M{20000}, K{10000}, N{20000};
    std::cout << "Creating matrices..." << std::endl;
    RowMatrix matA(M, K), matB(K, N);

    std::cout << "Filling matrices... " << std::endl;
    populateMatrix(matA);
    populateMatrix(matB);

    // std::cout << "CPU Matmul... " << std::endl;
    // auto start = std::chrono::high_resolution_clock::now();
    // RowMatrix cpuResult{matA * matB};
    // reportTime(start);

    MatMulImpl implId{MatMulImpl::Naive};
    RowMatrix gpuResult{matmulCuda(matA, matB, implId)};

    implId = MatMulImpl::SharedMem;
    gpuResult = matmulCuda(matA, matB, implId);

    implId = MatMulImpl::cuBLAS;
    gpuResult = matmulCuda(matA, matB, implId);

    // float tolerance{0.0001f};
    // checkMatrixApproxEquality(cpuResult, gpuResult, tolerance);
}
