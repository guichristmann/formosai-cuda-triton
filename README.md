# FormosAI -- Introduction to Kernel Programming with CUDA C++ and Triton

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

---

# Outline for Meeting Aug 6.

## CUDA C++

Basics -- Hello World.
    * Kernel Launch.
    * Programming model (Perspective of a Single Thread).
VecAdd
    * Blocks and Threads.
    * Compute Sanitizer.
Matmul
    * Implement Simple/Naive matmul.
    * Let's measure performance (ncu) -- TFLOPS
    * Implement Shared Memory matmul.
    * Run cuBLAS matmul.

## Triton

Basics + Hello World.
    * Kernel launch.
    * Programming model (Perspective of a block, vectorized).
    * Triton Interpret for debugging.
VecAdd
Matmul
    * Tiled.
    * Swizzling.
Benchmark - TFLOPS
    
