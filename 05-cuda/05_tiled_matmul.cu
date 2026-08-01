#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

constexpr int TILE = 16;

__global__ void matmul_naive(const float* A, const float* B, float* C, int n) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < n && col < n) {
    float acc = 0.0f;
    for (int k = 0; k < n; ++k) acc += A[row * n + k] * B[k * n + col];
    C[row * n + col] = acc;
  }
}

__global__ void matmul_tiled(const float* A, const float* B, float* C, int n) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];
  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;
  float acc = 0.0f;
  for (int t = 0; t < (n + TILE - 1) / TILE; ++t) {
    int ak = t * TILE + threadIdx.x;
    int bk = t * TILE + threadIdx.y;
    As[threadIdx.y][threadIdx.x] = (row < n && ak < n) ? A[row * n + ak] : 0.0f;
    Bs[threadIdx.y][threadIdx.x] = (bk < n && col < n) ? B[bk * n + col] : 0.0f;
    __syncthreads();
    for (int k = 0; k < TILE; ++k) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    __syncthreads();
  }
  if (row < n && col < n) C[row * n + col] = acc;
}

using Kernel = void (*)(const float*, const float*, float*, int);

float time_kernel(Kernel kernel, const float* A, const float* B, float* C, int n) {
  dim3 threads(TILE, TILE);
  dim3 blocks((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  kernel<<<blocks, threads>>>(A, B, C, n);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  return ms;
}

int main() {
  const int n = 1024;
  const size_t bytes = n * n * sizeof(float);
  float *A, *B, *C1, *C2;
  cudaMallocManaged(&A, bytes);
  cudaMallocManaged(&B, bytes);
  cudaMallocManaged(&C1, bytes);
  cudaMallocManaged(&C2, bytes);
  for (int i = 0; i < n * n; ++i) {
    A[i] = rand() / (float)RAND_MAX;
    B[i] = rand() / (float)RAND_MAX;
  }
  float naive = time_kernel(matmul_naive, A, B, C1, n);
  float tiled = time_kernel(matmul_tiled, A, B, C2, n);
  float max_diff = 0.0f;
  for (int i = 0; i < n * n; ++i) max_diff = fmaxf(max_diff, fabsf(C1[i] - C2[i]));
  double flops = 2.0 * n * n * n;
  std::printf("naive %.2f ms (%.0f GFLOP/s), tiled %.2f ms (%.0f GFLOP/s), max diff %.2e\n", naive,
              flops / naive / 1e6, tiled, flops / tiled / 1e6, max_diff);
  cudaFree(A);
  cudaFree(B);
  cudaFree(C1);
  cudaFree(C2);
  return 0;
}
