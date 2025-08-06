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

__global__ void simple_matmul(float* a, float* b, float* c, int64_t m, int64_t k, int64_t n)
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
    // TODO: Assuming block size is 32.
    constexpr std::size_t blockSize{32};

    // TODO: Dynamically allocate shared memory.
    // TODO: Remove assumption of hardcoded block size.
    __shared__ float shm_a[blockSize][blockSize];
    __shared__ float shm_b[blockSize][blockSize];

    int64_t col{blockIdx.x * blockDim.x + threadIdx.x};
    int64_t row{blockIdx.y * blockDim.y + threadIdx.y};

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

    // NOTE: Our SHM kernel has a hardcoded blocksize of 32, 32
    dim3 blockSize(32, 32);
    dim3 blocks{static_cast<uint32_t>((N + blockSize.y - 1) / blockSize.y),
                static_cast<uint32_t>((M + blockSize.x - 1) / blockSize.x)};

    auto tKernel{std::chrono::high_resolution_clock::now()};
    switch (impl)
    {
        case MatMulImpl::Naive:
        {
            simple_matmul<<<blocks, blockSize>>>(gpuMatA, gpuMatB, gpuMatResult, matA.rows(), matA.cols(), matB.cols());
            break;
        }
        case MatMulImpl::SharedMem:
        {
            matmul_shm<<<blocks, blockSize>>>(gpuMatA, gpuMatB, gpuMatResult, matA.rows(), matA.cols(), matB.cols());
            break;
        }
        case MatMulImpl::cuBLAS:
        {
            cublasHandle_t handle;
            CUBLAS_CHECK(cublasCreate(&handle));
            // cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH); // NOTE: Allow the use of Tensor Cores for FP32.
            const float alpha = 1.0f;
            const float beta = 0.0f;
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
    constexpr std::size_t M{4096}, K{8192}, N{4096};

    std::cout << "Creating matrices..." << std::endl;
    RowMatrix matA(M, K), matB(K, N);

    std::cout << "Filling matrices... " << std::endl;
    populateMatrix(matA);
    populateMatrix(matB);

    // CPU takes about 20 seconds.
    // std::cout << "CPU Matmul... " << std::endl;
    // auto start = std::chrono::high_resolution_clock::now();
    // RowMatrix cpuResult{matA * matB};
    // reportTime(start);

    std::cout << "#### Naive implementation ####\n";
    MatMulImpl implId{MatMulImpl::Naive};
    for (std::size_t i{0}; i < 10; ++i)
    {
        std::cout << "Run " << i + 1 << "... ";
        RowMatrix gpuResult{matmulCuda(matA, matB, implId)};
    }

    implId = MatMulImpl::SharedMem;
    std::cout << "#### SharedMem Implementation ####\n";
    for (std::size_t i{0}; i < 10; ++i)
    {
        std::cout << "Run " << i + 1 << "... ";
        RowMatrix gpuResult2{matmulCuda(matA, matB, implId)};
    }

    implId = MatMulImpl::cuBLAS;
    std::cout << "#### cuBLAS Implementation ####\n";
    for (std::size_t i{0}; i < 10; ++i)
    {
        std::cout << "Run " << i + 1 << "... ";
        RowMatrix gpuResult3{matmulCuda(matA, matB, implId)};
    }

    // NOTE: Use this to compare result between CPU and GPU mats.
    // float tolerance{0.0001f};
    // checkMatrixApproxEquality(cpuResult, gpuResult, tolerance);
}
