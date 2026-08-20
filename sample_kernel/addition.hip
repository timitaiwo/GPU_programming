#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include <hip/hip_runtime.h>

template <typename T> __global__ void square(T *A, const int64_t max) {

  int start = blockIdx.x * blockDim.x + threadIdx.x;

  for (int i = start; i < max && i < ; i++) {
    A[i] = A[i] * A[i];
  }
}

int main() {

  // Setup randomization
  std::random_device rand;
  std::mt19937 generator(rand());
  std::uniform_int_distribution<> distribution(1, 100);

  // Generate the vector
  const int MAX = 200;
  auto baseVec = std::vector<int32_t>(MAX);
  auto resVec = std::vector<int32_t>(MAX);

  for (auto &val : baseVec) {
    val = distribution(generator);
  }

  // Transfer the data to the GPU
  int *a_dev;
  const int numBytes = baseVec.size() * sizeof(baseVec[0]);

  auto hipState = hipMalloc(&a_dev, numBytes);
  if (hipState != hipSuccess) {
    std::cerr << "Memory allocation on GPU failed with error: "
              << hipGetErrorString(hipState) << std::endl;
    return hipState;
  }

  hipState = hipMemcpy(a_dev, baseVec.data(), numBytes, hipMemcpyHostToDevice);
  if (hipState != hipSuccess) {
    std::cerr << "Memory allocation on GPU failedwith error: "
              << hipGetErrorString(hipState) << std::endl;
    return hipState;
  }

  // Launch kernel
  square<<<1, 1>>>(a_dev, baseVec.size());

  // Synchronize i.e. wait for kernel to finish
  hipState = hipDeviceSynchronize();
  if (hipState != hipSuccess) {
    std::cerr << "GPU failed to synchronize after kernel launch with error: "
              << hipGetErrorString(hipState) << std::endl;
    return hipState;
  }

  // Extract the squared vector from the GPU
  hipState = hipMemcpy(resVec.data(), a_dev, numBytes, hipMemcpyDeviceToHost);
  if (hipState != hipSuccess) {
    std::cerr << "Result from GPU not extracted with error: "
              << hipGetErrorString(hipState) << std::endl;
    return hipState;
  }

  // Compare resVec and baseVec values
  bool expected = true;
  for (int i = 0; i < baseVec.size(); i++) {
    if (baseVec[i] * baseVec[i] != resVec[i]) {
      std::cerr << i << " with base value " << baseVec[i]
                << " is not half of res value " << resVec[i] << std::endl;
      expected = false;
    }
  }

  if (expected == true) {
    std::cout << "Kernel executed successfully" << std::endl;
  }

  hipState = hipFree(
      a_dev); // hipFree marked as nodiscard so best to assign it to something
  return 0;
}
