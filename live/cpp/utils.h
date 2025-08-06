#include <chrono>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <string>
#include <vector>

using RowMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using TimePoint = std::chrono::high_resolution_clock::time_point;

#define cudaCheckErrors(msg)                                                                                        \
    do                                                                                                              \
    {                                                                                                               \
        cudaError_t __err = cudaGetLastError();                                                                     \
        if (__err != cudaSuccess)                                                                                   \
        {                                                                                                           \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", msg, cudaGetErrorString(__err), __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n");                                                             \
            exit(1);                                                                                                \
        }                                                                                                           \
    } while (0)

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

inline void reportTimeUntilNow(std::chrono::high_resolution_clock::time_point start, const std::string& identifier)
{
    using namespace std::chrono;
    using namespace std::chrono_literals;
    auto now{high_resolution_clock::now()};

    int64_t durationUs{duration_cast<microseconds>(now - start).count()};

    std::cout << "[" << identifier << "]: Took " << durationUs << " us.\n";
}

template <typename T>
bool areVectorsApproximatelyEqual(const std::vector<T>& vec1, const std::vector<T>& vec2, T tolerance)
{
    if (vec1.size() != vec2.size())
    {
        std::cout << "Vector sizes differ: vec1.size() = " << vec1.size() << ", vec2.size() = " << vec2.size()
                  << std::endl;
        return false;
    }

    for (size_t i = 0; i < vec1.size(); ++i)
    {
        if (std::abs(vec1[i] - vec2[i]) > tolerance)
        {
            std::cout << "Elements at index " << i << " differ by more than tolerance."
                      << " vec1[" << i << "] = " << vec1[i] << ", vec2[" << i << "] = " << vec2[i]
                      << ", difference = " << std::abs(vec1[i] - vec2[i]) << std::endl;
            return false;
        }
    }

    return true;
}

/**
 * @brief Checks if elements of two RowMatrix objects are approximately equal
 * within a given tolerance and reports any differing elements.
 *
 * This function calculates the absolute difference between corresponding elements
 * of two input matrices. If the absolute difference for any element exceeds the
 * specified tolerance, it prints the coordinates (row, column), the values from
 * both matrices, and their absolute difference.
 *
 * @param mat1 The first RowMatrix object to compare.
 * @param mat2 The second RowMatrix object to compare.
 * @param tolerance The maximum allowed absolute difference for elements to be
 * considered approximately equal (fp32 tolerance).
 */
void checkMatrixApproxEquality(const RowMatrix& mat1, const RowMatrix& mat2, float tolerance)
{
    // First, check if the dimensions of the matrices match.
    // Comparing matrices with different dimensions is nonsensical for element-wise checks.
    if (mat1.rows() != mat2.rows() || mat1.cols() != mat2.cols())
    {
        std::cerr << "Error: Matrices have different dimensions. Cannot perform element-wise comparison." << std::endl;
        std::cerr << "  Matrix 1 dimensions: " << mat1.rows() << "x" << mat1.cols() << std::endl;
        std::cerr << "  Matrix 2 dimensions: " << mat2.rows() << "x" << mat2.cols() << std::endl;
        return;  // Exit the function if dimensions don't match.
    }

    // Calculate the absolute element-wise difference between the two matrices.
    // We convert to Array for element-wise operations like cwiseAbs().
    RowMatrix diff_matrix = (mat1 - mat2).cwiseAbs();

    // Create a boolean matrix where 'true' indicates an element's difference
    // exceeds the given tolerance.
    Eigen::ArrayXX<bool> exceeds_tolerance = (diff_matrix.array() > tolerance);

    std::cout << "--- Checking Matrix Approximate Equality (Tolerance: " << tolerance << ") ---\n";
    std::cout << "Elements differing by more than the tolerance:\n";

    bool found_differences = false;
    // Iterate through each element of the boolean matrix.
    // If exceeds_tolerance(i, j) is true, it means mat1(i, j) and mat2(i, j)
    // are not approximately equal.
    for (int i = 0; i < exceeds_tolerance.rows(); ++i)
    {
        for (int j = 0; j < exceeds_tolerance.cols(); ++j)
        {
            if (exceeds_tolerance(i, j))
            {
                found_differences = true;
                std::cout << "  Position (" << i << ", " << j << "):\n";
                std::cout << "    Matrix 1 value: " << mat1(i, j) << "\n";
                std::cout << "    Matrix 2 value: " << mat2(i, j) << "\n";
                std::cout << "    Absolute difference: " << diff_matrix(i, j) << "\n";
            }
        }
    }

    if (!found_differences)
    {
        std::cout << "  All elements are approximately equal within the specified tolerance." << std::endl;
    }
    std::cout << "-------------------------------------------------------------------\n";
}

inline void reportTime(TimePoint t)
{
    auto now{std::chrono::high_resolution_clock::now()};

    int64_t durationUs{std::chrono::duration_cast<std::chrono::microseconds>(now - t).count()};

    std::cout << "Took " << static_cast<double>(durationUs) / 1000.0 << " ms\n";
}

inline void populateMatrix(RowMatrix& m)
{
    for (std::size_t i{0}; i < m.rows(); ++i)
    {
        for (std::size_t j{0}; j < m.cols(); ++j)
        {
            m(i, j) = static_cast<float>(rand() % 1000 - 500) / 1000.0;
        }
    }
}
