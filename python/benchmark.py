import triton
import torch

from matmul_triton import triton_matmul

DEVICE = "cuda:0"
DTYPE = torch.float32

configs = [
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],
        x_vals=[128 * i for i in range(2, 35)],
        line_arg="method",
        line_vals=["triton", "torch_default", "torch_medium"],
        line_names=["Triton", "Torch (Highest Precision)", "Torch (Medium Precision)"],
        styles=[("green", "-"), ("blue", "-"), ("lightblue", "-")],
        ylabel="TFLOPS",
        plot_name=f"Matmul Performance {DTYPE}",
        args={},
    )
]

@triton.testing.perf_report(configs)
def benchmark(M, N, K, method):
    A = torch.rand((M, K), dtype=DTYPE, device=DEVICE) - 0.5
    B = torch.rand((K, N), dtype=DTYPE, device=DEVICE) - 0.5

    quantiles = [0.5, 0.2, 0.8]
    if method == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_matmul(A, B, todtype=DTYPE), quantiles=quantiles) # type: ignore
    if method == "torch_default":
        torch.set_float32_matmul_precision("highest")
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(A, B), quantiles=quantiles) # type: ignore
    if method == "torch_medium":
        torch.set_float32_matmul_precision("medium")
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(A, B), quantiles=quantiles) # type: ignore

    number_of_ops = 2 * M * K * N
    time_to_tflops = lambda time_ms: number_of_ops * 1e-12 / (time_ms * 1e-3)

    return time_to_tflops(ms), time_to_tflops(max_ms), time_to_tflops(min_ms) # type: ignore

benchmark.run(print_data=True)
