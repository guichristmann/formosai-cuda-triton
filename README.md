# FormosAI -- Introduction to Kernel Programming with CUDA C++ and Triton

## Outline for Meeting Aug 6.

**CUDA C++**

* Basics -- Hello World.
    * Kernel Launch.
    * Programming model (Perspective of a Single Thread).
* VecAdd
    * Blocks and Threads.
    * Compute Sanitizer.
* Matmul
    * Implement Simple/Naive matmul.
    * Let's measure performance (ncu) -- TFLOPS
    * Implement Shared Memory matmul.
    * Run cuBLAS matmul.

**Triton**

* Basics + Hello World.
    * Kernel launch.
    * Programming model (Perspective of a block, vectorized).
    * Triton Interpret for debugging.
* VecAdd
* Matmul
    * Tiled.
    * Swizzling.
* Benchmark - TFLOPS

---
## Examples

```
├── cpp -- Kernels written with CUDA C++.
│   ├── matmul.cu
│   ├── vecadd.cu
│   └── utils.h
├── python -- Kernels with Triton and PyTorch code.
│   ├── benchmark.py
│   ├── matmul_torch.py
│   ├── matmul_triton.py
│   ├── vecadd_torch.py
│   ├── vecadd_triton.py
│   ├── requirements.txt
│   └── utils.py
```

---

### Learning Resources and Interesting Links
* [Simplest CUDA Introduction](https://developer.nvidia.com/blog/easy-introduction-cuda-c-and-c/)
* [CUDA Training Series](https://www.olcf.ornl.gov/cuda-training-series/)
    * [My notes, up to 5th video/class](https://sunset-cloud-67f.notion.site/CUDA-Training-Series-Notes-233c6cd8463f81858db1d4e3b7b2d5fa)
* [Simon Boehm - How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM)

* [Official Triton Docs](https://triton-lang.org/main/index.html)
* [Discord Group - GPU Mode](https://discord.com/invite/gpumode)
* [A Practitioner's Guide to Triton](https://github.com/gpu-mode/lectures/blob/main/lecture_014/A_Practitioners_Guide_to_Triton.ipynb)
* [Triton Puzzles](https://github.com/srush/Triton-Puzzles?tab=readme-ov-file)
