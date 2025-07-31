import torch
from time import perf_counter_ns

M = 20000
K = 10000
N = 20000

device = "cuda:0"

torch.softmax

# Warmup...
for i in range(5):
    A = torch.rand((M, K), dtype=torch.float32, device=device)
    B = torch.rand((K, N), dtype=torch.float32, device=device)
    C = A.matmul(B)
    torch.cuda.synchronize()

# Measuring
for i in range(5):
    A = torch.rand((M, K), dtype=torch.float32, device=device)
    B = torch.rand((K, N), dtype=torch.float32, device=device)
    _t_start = perf_counter_ns()

    C = A.matmul(B)
    torch.cuda.synchronize()

    elapsed_ns = perf_counter_ns() - _t_start

    ms = elapsed_ns / 1e6
    print(f"Matmul took {ms:.2f} milliseconds.")
