import torch
from time import perf_counter_ns

M = 4096
K = 8192
N = 4096

device = "cuda:0"
# NOTE: Switching this to float16 uses Tensor Core matmuls.
dtype = torch.float32
# NOTE:: Or this when using float32:  
# https://docs.pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
# torch.set_float32_matmul_precision("medium")

# Warmup...
for i in range(5):
    A = torch.rand((M, K), dtype=torch.float32, device=device)
    B = torch.rand((K, N), dtype=torch.float32, device=device)
    C = A.matmul(B)
    torch.cuda.synchronize()

# Measuring
for i in range(10):
    A = torch.rand((M, K), dtype=torch.float32, device=device)
    B = torch.rand((K, N), dtype=torch.float32, device=device)
    _t_start = perf_counter_ns()

    C = A.matmul(B)
    torch.cuda.synchronize()

    elapsed_ns = perf_counter_ns() - _t_start

    ms = elapsed_ns / 1e6
    print(f"Matmul took {ms:.2f} milliseconds.")
