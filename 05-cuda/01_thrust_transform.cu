#include <cstdio>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

struct Saxpy {
  float a;
  __host__ __device__ float operator()(float x, float y) const { return a * x + y; }
};

int main() {
  const int n = 1 << 24;
  thrust::device_vector<float> x(n, 1.0f), y(n, 2.0f);
  thrust::transform(thrust::device, x.begin(), x.end(), y.begin(), y.begin(), Saxpy{3.0f});
  float sum = thrust::reduce(thrust::device, y.begin(), y.end(), 0.0f);
  std::printf("sum = %.0f (expected %.0f)\n", sum, 5.0f * n);
  return 0;
}
