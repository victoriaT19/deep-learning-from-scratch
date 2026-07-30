#include <cstdio>
#include <cuda_runtime.h>

template <int BLOCK>
__global__ void block_sum(const float* in, float* out, int n) {
  __shared__ float s[BLOCK];
  int tid = threadIdx.x;
  float v = 0.0f;
  for (int i = blockIdx.x * BLOCK + tid; i < n; i += BLOCK * gridDim.x) v += in[i];
  s[tid] = v;
  __syncthreads();
  for (int stride = BLOCK / 2; stride > 32; stride >>= 1) {
    if (tid < stride) s[tid] += s[tid + stride];
    __syncthreads();
  }
  if (tid < 32) {
    v = s[tid] + s[tid + 32];
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffff, v, offset);
    if (tid == 0) atomicAdd(out, v);
  }
}

int main() {
  const int n = 1 << 24;
  float *x, *sum;
  cudaMallocManaged(&x, n * sizeof(float));
  cudaMallocManaged(&sum, sizeof(float));
  for (int i = 0; i < n; ++i) x[i] = 1.0f;
  *sum = 0.0f;
  block_sum<256><<<1024, 256>>>(x, sum, n);
  cudaDeviceSynchronize();
  std::printf("sum = %.0f (expected %d)\n", *sum, n);
  cudaFree(x);
  cudaFree(sum);
  return 0;
}
