import os

os.environ["TRITON_INTERPRET"] = "0"

import triton
import triton.language as tl
import torch
import numpy as np
from time import perf_counter_ns
import utils

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)

# TODO: Add tuning of Group Size and other block sies.
@triton.autotune(
    configs=[
        triton.Config({"BS": 64}, num_stages=5, num_warps=4),
        triton.Config({"BS": 64}, num_stages=5, num_warps=8),
        triton.Config({"BS": 64}, num_stages=4, num_warps=4),
        triton.Config({"BS": 64}, num_stages=4, num_warps=8),
        triton.Config({"BS": 64}, num_stages=3, num_warps=4),
        triton.Config({"BS": 64}, num_stages=3, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def tiled_matmul_kernel(
    ptr_a,
    ptr_b,
    M,
    K,
    N,
    ptr_result,
    BS: tl.constexpr,
    group_sz: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    num_pid_m, num_pid_n = tl.num_programs(0), tl.num_programs(1)
    pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, group_sz)

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    m_offs = pid_m * BS + tl.arange(0, BS)  # "Fixed"
    n_offs = pid_n * BS + tl.arange(0, BS)  # "Fixed"
    k_offs = tl.arange(0, BS)  # Incremented over loop.

    # We can compute the M and N masks pre-loop, since they are fixed throughout
    # the program.
    mask_m = m_offs < M
    mask_n = n_offs < N

    block_acc = tl.zeros((BS, BS), dtype=tl.float32)
    for i in range(0, K, BS):
        # ~TODO~: Any benefit moving these pre-loop and doing only the increment in-loop?
        # I'd expect that the compiler could sort out such as simple optimization.
        # R: Measured no significant difference, and I find this style a bit easier to read.
        a_offs_2d = m_offs[:, None] * K + k_offs[None, :]
        b_offs_2d = k_offs[:, None] * N + n_offs[None, :]

        mask_k = k_offs < K

        mask_a = mask_m[:, None] & mask_k[None, :]
        mask_b = mask_k[:, None] & mask_n[None, :]

        data_a = tl.load(ptr_a + a_offs_2d, mask=mask_a, other=0.0)
        data_b = tl.load(ptr_b + b_offs_2d, mask=mask_b, other=0.0)

        # NOTE: `tl.dot` only works for dims > 16. Won't compile otherwise.
        # NOTE: `input_precision` by default is `tf32`. TF32 is the 32-bit floating point
        # used by Tensor Cores.
        block_acc += tl.dot(data_a, data_b, input_precision="ieee")

        k_offs += BS

    # Apply an activation function on the accumulator.
    # block_acc = leaky_relu(block_acc)

    c_offs_2d = m_offs[:, None] * N + n_offs[None, :]
    mask_c = mask_m[:, None] & mask_n[None, :]
    tl.store(ptr_result + c_offs_2d, block_acc, mask=mask_c)


# @triton.jit
# def bad_matmul_kernel(
#     ptr_a,
#     ptr_b,
#     M: tl.constexpr,
#     K: tl.constexpr,
#     N: tl.constexpr,
#     ptr_result,
#     BS: tl.constexpr,
# ):
#     """
#     This is a pretty terrible kernel that I first implemented after coming from CUDA C++, and 
#     before reading/understanding better the Triton programming model. 
#
#     In this kernel each program computes the ouput for a single element in C, by multiplying a 
#     single row and column from A and B. It is much less efficient and also requires a reduce
#     sum operation (`c = tl.sum(rowdata_a * coldata_a`).
#
#     I basically forcefully applied the CUDA programming model/way of thinking with Triton.
#
#     I'll leave it here an example of what NOT to do when implementing a kernel with Triton ;)
#
#     """
#     pid_ax0 = tl.program_id(axis=0)
#     pid_ax1 = tl.program_id(axis=1)
#
#     numel_a = M * K
#     numel_b = K * N
#     numel_c = M * N
#     k_offsets = tl.arange(0, triton.next_power_of_2(K))  # type: ignore[ignoreArgumentType]
#     mask_k_offsets = k_offsets < K
#
#     for i in range(BS):
#         row_offs = k_offsets + i * K
#         row_offs += pid_ax0 * BS * K
#         row_mask = (row_offs < numel_a) & (mask_k_offsets)
#         rowdata_a = tl.load(ptr_a + row_offs, mask=row_mask)
#
#         for j in range(BS):
#             col_offs = k_offsets * N + j + pid_ax1 * BS
#             col_mask = (k_offsets * 0 + j + pid_ax1 * BS) < N
#
#             coldata_b = tl.load(ptr_b + col_offs, mask=col_mask & mask_k_offsets)
#
#             c = tl.sum(rowdata_a * coldata_b)
#             out_row = i + pid_ax0 * BS
#             out_col = j + pid_ax1 * BS
#
#             out_index = out_row * N + out_col
#             out_mask = (out_row < M) & (out_col < N)
#             tl.store(ptr_result + out_index, c, mask=out_mask)


def triton_matmul(
    tensor_a: torch.Tensor, tensor_b: torch.Tensor, block_size: int = 16, todtype: torch.dtype | None = None
) -> torch.Tensor:
    original_dtype = tensor_a.dtype
    assert tensor_a.dtype == tensor_b.dtype

    if todtype:
        tensor_a = tensor_a.to(todtype)
        tensor_b = tensor_b.to(todtype)

    tensor_result = torch.zeros(
        size=(tensor_a.shape[0], tensor_b.shape[1]), dtype=todtype, device=DEVICE
    )

    M = tensor_a.shape[0]
    K = tensor_a.shape[1]
    N = tensor_b.shape[1]

    grid = lambda META: (triton.cdiv(M, META["BS"]), triton.cdiv(N, META["BS"]), )
    tiled_matmul_kernel[grid](
        tensor_a,
        tensor_b,
        M,  # type:ignore[reportArgumentType]
        K,  # type:ignore[reportArgumentType]
        N,  # type:ignore[reportArgumentType]
        tensor_result,
        # BS=block_size,  # type: ignore
        group_sz=16,
    )

    return tensor_result.to(original_dtype)


def fill_values_with_inds(t: torch.Tensor) -> None:
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            t[i, j] = i * t.shape[1] + j


if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=2)

    M = 4096
    K = 8192
    N = 4096

    torch.manual_seed(2)

    # NOTE: If our distribution is biased towards positive/negative end, we can
    # have precision problems during accumulation. Also if the scale of numbers is large.
    # I'm using [-0.5, 0.5]
    A = torch.rand((M, K), dtype=torch.float32, device=DEVICE) - 0.5
    B = torch.rand((K, N), dtype=torch.float32, device=DEVICE) - 0.5

    C = A.matmul(B)
    print("A")
    print(A.cpu().numpy())
    print("B")
    print(B.cpu().numpy())

    triton_C = triton_matmul(A, B)
    torch.cuda.synchronize()
    np.set_printoptions(suppress=True, precision=2)
    # NOTE: Require a largeish tolerance to consider a match.
    closeenough = torch.allclose(C, triton_C, rtol=0.0, atol=0.05)
    print(f"Are they close?", closeenough)

    # Warmup
    for i in range(10):
        triton_C = triton_matmul(A, B)
    torch.cuda.synchronize()

    # Measure
    for i in range(10):
        t_start = perf_counter_ns()
        triton_C = triton_matmul(A, B)
        torch.cuda.synchronize()
        utils.reportTimeUntilNow(t_start, f"Triton")
