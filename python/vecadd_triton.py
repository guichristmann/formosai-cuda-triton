import torch
from time import perf_counter_ns
import utils

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Mask to set out-of-bounds elements to `False`.
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y

    tl.store(out_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor, block_size: int = 1024):
    output = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    n_elements = output.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    # Launch kernel.
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=block_size)  # type: ignore

    return output


if __name__ == "__main__":
    torch.manual_seed(0)
    DSIZE = 256 * 1024 * 1024
    x = torch.rand(DSIZE, device=DEVICE)
    y = torch.rand(DSIZE, device=DEVICE)

    #### Torch for reference ####

    # Warmup...
    for i in range(10):
        output_torch = x + y
    torch.cuda.synchronize()

    for i in range(5):
        t_start = perf_counter_ns()
        output_torch = x + y
        torch.cuda.synchronize()
        utils.reportTimeUntilNow(t_start, "Torch")

    #### Triton ####
    for bs in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        # Warmup...
        for i in range(10):
            output_triton = add(x, y, bs)
        torch.cuda.synchronize()

        t_start = perf_counter_ns()
        output_triton = add(x, y, bs)
        torch.cuda.synchronize()
        utils.reportTimeUntilNow(t_start, f"Triton @ {bs}")
