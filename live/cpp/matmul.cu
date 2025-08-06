#include <eigen3/Eigen/Dense>

#include "utils.h"

constexpr uint32_t M{4}, K{8}, N{4};

RowMatrix gpuMatmul(const RowMatrix& A, const RowMatrix& B)
{
    // TODO: Write our matmul kernel.
    return A * B;
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
