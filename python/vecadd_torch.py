import torch
from time import perf_counter_ns
import utils

DSIZE = 256 * 1024 * 1024

a = torch.ones((DSIZE,), dtype=torch.float32) * 1.0
b = torch.ones((DSIZE,), dtype=torch.float32) * 2.0

a = a.to("cpu")
b = b.to("cpu")

t_start = perf_counter_ns()
c = a + b
torch.cuda.synchronize()
utils.reportTimeUntilNow(t_start, "CPU Add")

a = a.to("cuda:0")
b = b.to("cuda:0")
c = torch.empty_like(a, device="cuda:0")

# Warmup...
for i in range(50):
    c = a + b
# If not added here, can mess with the timing of the benchmark below.
torch.cuda.synchronize()

# Measure time...
for i in range(5):
    t_start = perf_counter_ns()
    c = a + b
    torch.cuda.synchronize()
    utils.reportTimeUntilNow(t_start, "GPU Add")
