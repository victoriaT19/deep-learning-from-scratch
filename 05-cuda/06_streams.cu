#include <cstdio>
#include <cuda_runtime.h>

__global__ void work(float* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    for (int k = 0; k < 100; ++k) x[i] = x[i] * 0.999f + 0.001f;
}

int main() {
  const int n = 1 << 24;
  const int chunks = 4;
  const int chunk = n / chunks;
  float *h, *d;
  cudaMallocHost(&h, n * sizeof(float));
  cudaMalloc(&d, n * sizeof(float));
  for (int i = 0; i < n; ++i) h[i] = 1.0f;
  cudaStream_t streams[chunks];
  for (auto& s : streams) cudaStreamCreate(&s);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  for (int c = 0; c < chunks; ++c) {
    int off = c * chunk;
    cudaMemcpyAsync(d + off, h + off, chunk * sizeof(float), cudaMemcpyHostToDevice, streams[c]);
    work<<<(chunk + 255) / 256, 256, 0, streams[c]>>>(d + off, chunk);
    cudaMemcpyAsync(h + off, d + off, chunk * sizeof(float), cudaMemcpyDeviceToHost, streams[c]);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  std::printf("%d streams overlapping copy and compute: %.2f ms, h[0] = %.4f\n", chunks, ms, h[0]);
  for (auto& s : streams) cudaStreamDestroy(s);
  cudaFreeHost(h);
  cudaFree(d);
  return 0;
}
