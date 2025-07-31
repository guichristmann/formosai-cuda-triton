import os

os.environ["TRITON_INTERPRET"] = "0"

import triton
import triton.language as tl
import torch
import numpy as np
from time import perf_counter_ns
import utils

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# @triton.autotune(
#     configs=[
#         triton.Config({"BS": 128}, num_warps=4),
#         triton.Config({"BS": 256}, num_warps=4),
#         triton.Config({"BS": 512}, num_warps=4),
#         triton.Config({"BS": 1024}, num_warps=4),
#         triton.Config({"BS": 128}, num_warps=8),
#         triton.Config({"BS": 256}, num_warps=8),
#         triton.Config({"BS": 512}, num_warps=8),
#         triton.Config({"BS": 1024}, num_warps=8),
#     ],
#     key=["M", "K", "N"]
# )
@triton.jit
def matmul_kernel(
    ptr_a,
    ptr_b,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    ptr_result,
    BS: tl.constexpr,
):
    pid_ax0 = tl.program_id(axis=0)
    pid_ax1 = tl.program_id(axis=1)

    numel_a = M * K
    numel_b = K * N
    numel_c = M * N
    k_offsets = tl.arange(0, triton.next_power_of_2(K))  # type: ignore[ignoreArgumentType]
    mask_k_offsets = k_offsets < K

    for i in range(BS):
        row_offs = k_offsets + i * K
        row_offs += pid_ax0 * BS * K
        row_mask = (row_offs < numel_a) & (mask_k_offsets)
        rowdata_a = tl.load(ptr_a + row_offs, mask=row_mask)

        for j in range(BS):
            col_offs = k_offsets * N + j + pid_ax1 * BS
            col_mask = (k_offsets * 0 + j + pid_ax1 * BS) < N

            coldata_b = tl.load(ptr_b + col_offs, mask=col_mask & mask_k_offsets)

            c = tl.sum(rowdata_a * coldata_b)
            out_row = i + pid_ax0 * BS
            out_col = j + pid_ax1 * BS

            out_index = out_row * N + out_col
            out_mask = (out_row < M) & (out_col < N)
            tl.store(ptr_result + out_index, c, mask=out_mask)


def triton_matmul(
    tensor_a: torch.Tensor, tensor_b: torch.Tensor, block_size: int = 8
) -> torch.Tensor:
    dtype = torch.float32

    tensor_a = tensor_a.to(dtype)
    tensor_b = tensor_b.to(dtype)

    tensor_result = torch.zeros(
        size=(tensor_a.shape[0], tensor_b.shape[1]), dtype=dtype, device=DEVICE
    )

    M = tensor_a.shape[0]
    K = tensor_a.shape[1]
    N = tensor_b.shape[1]

    i_blocks = triton.cdiv(M, block_size)
    j_blocks = triton.cdiv(N, block_size)
    # grid = lambda META: (triton.cdiv(M, META["BS"]), triton.cdiv(N, META["BS"]))
    matmul_kernel[(i_blocks, j_blocks)](
        tensor_a,
        tensor_b,
        M,  # type:ignore[reportArgumentType]
        K,  # type:ignore[reportArgumentType]
        N,  # type:ignore[reportArgumentType]
        tensor_result,
        BS=block_size,
    )

    return tensor_result


if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=2)

    # Works for these.
    M = 500
    K = 100
    N = 200

    torch.manual_seed(2)
    # NOTE: If our distribution is not roughly symmetrical in the negative range
    A = torch.rand((M, K), dtype=torch.float32, device=DEVICE) * 2.0 - 0.5
    B = torch.rand((K, N), dtype=torch.float32, device=DEVICE) * 2.0 - 0.5
    C = A.matmul(B)
    print("A")
    print(A.cpu().numpy())
    print("B")
    print(B.cpu().numpy())

    triton_C = triton_matmul(A, B)
    torch.cuda.synchronize()
    np.set_printoptions(suppress=True, precision=2)
    closeenough = torch.allclose(C, triton_C, rtol=0.0, atol=0.001)
    print(f"Are they close?", closeenough)

    # Warmup
    # for i in range(10):
    #     triton_C = triton_matmul(A, B)
    # torch.cuda.synchronize()

    # Measure
    for i in range(100):
        t_start = perf_counter_ns()
        triton_C = triton_matmul(A, B)
        torch.cuda.synchronize()
        utils.reportTimeUntilNow(t_start, f"Triton")
