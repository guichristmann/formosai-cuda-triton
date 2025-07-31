#include <chrono>
#include <iostream>
#include <memory>
#include <vector>

#include "utils.h"

constexpr std::size_t DSIZE{256 * 1024 * 1024};

void cpuAdd(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& c)
{
    for (std::size_t i{0}; i < a.size(); ++i)
    {
        c[i] = a[i] + b[i];
    }
}

__global__ void kernel_vecAdd(float* a, float* b, float* c, std::size_t size)
{
    std::size_t idx{threadIdx.x + blockIdx.x * blockDim.x};

    if (idx < size)
    {
        c[idx] = a[idx] + b[idx];
    }
}

int main()
{
    // Creating and filling data in CPU.
    std::vector<float> vec1(DSIZE, 1.0f);
    std::vector<float> vec2(DSIZE, 2.0f);
    std::vector<float> cpuResult(DSIZE, 0.0f);

    // Computing result in CPU.
    auto tStart{std::chrono::high_resolution_clock::now()};
    cpuAdd(vec1, vec2, cpuResult);
    reportTimeUntilNow(tStart, "CPU Add");

    // Computing result in GPU.
    float *d_vec1, *d_vec2, *d_result;  // `d_` for "device", i.e. GPU.

    cudaMalloc(&d_vec1, sizeof(float) * DSIZE);
    cudaMalloc(&d_vec2, sizeof(float) * DSIZE);
    cudaMalloc(&d_result, sizeof(float) * DSIZE);
    cudaCheckErrors("Failed to malloc device pointers.");

    cudaMemcpy(d_vec1, vec1.data(), sizeof(float) * DSIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec2, vec2.data(), sizeof(float) * DSIZE, cudaMemcpyHostToDevice);
    cudaCheckErrors("Failed to copy vec data to gpu.");

	// Warmup...
    for (std::size_t i{1}; i <= 50; ++i)
    {
        int64_t nThreads{1024};
        tStart = std::chrono::high_resolution_clock::now();
        std::size_t nBlocks{(DSIZE + nThreads - 1) / nThreads};
        kernel_vecAdd<<<nBlocks, nThreads>>>(d_vec1, d_vec2, d_result, DSIZE);
        cudaDeviceSynchronize();
    }

	// Benchmark, report time.
    for (std::size_t nThreads{1}; nThreads <= 1024; nThreads = nThreads * 2)
    {
        // int64_t nThreads{1024};
        tStart = std::chrono::high_resolution_clock::now();
        std::size_t nBlocks{(DSIZE + nThreads - 1) / nThreads};
        kernel_vecAdd<<<nBlocks, nThreads>>>(d_vec1, d_vec2, d_result, DSIZE);
        cudaDeviceSynchronize();

		std::string msg{"GPU Add @ " + std::to_string(nThreads)};
        reportTimeUntilNow(tStart, msg);
    }

    // How to get this into a std::vector<float> in an idiomatic way?
    std::vector<float> gpuResult(DSIZE);
    cudaMemcpy(gpuResult.data(), d_result, sizeof(float) * DSIZE, cudaMemcpyDeviceToHost);

    std::cout << "Results approx. equal? " << areVectorsApproximatelyEqual(cpuResult, gpuResult, 0.001f) << "\n";
}
