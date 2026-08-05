# CUDA C++ — notes (NVIDIA Accelerated Computing Hub, cuda-cpp tutorials)

Build and run everything with `make run` on a machine with an NVIDIA GPU and the CUDA toolkit.

1. **Parallel algorithms first** (`01_thrust_transform.cu`): before writing kernels, express the computation with Thrust/CUB algorithms (`transform`, `reduce`) and an execution policy. The functor is marked `__host__ __device__` so it can run on either side.
2. **Kernels and the thread hierarchy** (`02_vector_add.cu`): a kernel launch `<<<blocks, threads>>>` creates a grid of blocks of threads; each thread computes its global index `blockIdx.x * blockDim.x + threadIdx.x` and guards against `i >= n`. Always check `cudaGetLastError()` after a launch and synchronize before reading results on the host.
3. **Grid-stride loops** (`03_grid_stride.cu`): launch a grid sized to the GPU (a multiple of the number of SMs) and let each thread stride over the array; the kernel then works for any `n`. Prefetching managed memory avoids page faults inside the timed region. This kernel is memory-bound, so the useful metric is GB/s, not FLOP/s.
4. **Shared memory and cooperation** (`04_reduction_shared.cu`): threads of a block share fast on-chip memory and synchronize with `__syncthreads()`. Tree reduction in shared memory, then warp shuffles (`__shfl_down_sync`) for the last 32 values, then one `atomicAdd` per block.
5. **Tiling** (`05_tiled_matmul.cu`): the naive matmul reads each input element `n` times from global memory; loading 16×16 tiles into shared memory reuses each load 16 times. Same results, much higher GFLOP/s. This is the idea behind the kernels PyTorch calls for `@`.
6. **Asynchrony and streams** (`06_streams.cu`): pinned host memory (`cudaMallocHost`) and several streams let copies and kernels of different chunks overlap.

Why I studied this before the LLM training part: it explains why batch size, sequence length and model width should be multiples of 8/64, why attention is memory-bound (hence FlashAttention), and what `torch.compile` and fused kernels actually save.
