#include <cstdio>
#include <cuda_runtime.h>

#define CHECK(call)                                              \
  do {                                                           \
    cudaError_t e = (call);                                      \
    if (e != cudaSuccess) {                                      \
      std::printf("CUDA error: %s\n", cudaGetErrorString(e));    \
      return 1;                                                  \
    }                                                            \
  } while (0)

__global__ void add(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

int main() {
  const int n = 1 << 20;
  const size_t bytes = n * sizeof(float);
  float *a, *b, *c;
  CHECK(cudaMallocManaged(&a, bytes));
  CHECK(cudaMallocManaged(&b, bytes));
  CHECK(cudaMallocManaged(&c, bytes));
  for (int i = 0; i < n; ++i) {
    a[i] = 1.0f;
    b[i] = 2.0f;
  }
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  add<<<blocks, threads>>>(a, b, c, n);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  int errors = 0;
  for (int i = 0; i < n; ++i) errors += c[i] != 3.0f;
  std::printf("%d blocks x %d threads, errors: %d\n", blocks, threads, errors);
  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
  return 0;
}
