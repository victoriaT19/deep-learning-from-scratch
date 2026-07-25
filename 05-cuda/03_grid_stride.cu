#include <cstdio>
#include <cuda_runtime.h>

__global__ void scale(float* x, float s, int n) {
  int stride = blockDim.x * gridDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) x[i] *= s;
}

int main() {
  const int n = 1 << 26;
  float* x;
  cudaMallocManaged(&x, n * sizeof(float));
  for (int i = 0; i < n; ++i) x[i] = 1.0f;
  int device, sms;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
  cudaMemPrefetchAsync(x, n * sizeof(float), device);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  scale<<<32 * sms, 256>>>(x, 2.0f, n);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  std::printf("%d SMs, %.3f ms, %.1f GB/s, x[n-1] = %.1f\n", sms, ms, 2.0 * n * sizeof(float) / ms / 1e6, x[n - 1]);
  cudaFree(x);
  return 0;
}
