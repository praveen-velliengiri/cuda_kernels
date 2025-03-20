//implement a simple reduction(sum) kernel
#include <iostream>
#include <cuda_runtime.h>

// Define constants
//#define N 9999
#define BLK_SIZE 128

using namespace std;
//each thread block is responsible for BLK_SIZE x 2 chunk of input data.
__global__ void simt_deadlock_without_yield() {
  __shared__ int flag;
  if (threadIdx.x == 0)
    flag = 0;
  __syncthreads();
  if (threadIdx.x % 2 == 0) {
    while (flag == 0);
    printf("\n wrap 2 set the flag");
  } else {
    flag = 1;
  }
}

int cpu_reduction_func(int *input, int len) {
  int sum = 0.0f;
  for (int i=0; i<len; i++) {
    sum += input[i];
  }
  return sum;
}

int main(int argc, char **argv) {

    simt_deadlock_without_yield<<<1, 32>>>();
    cudaDeviceSynchronize();

    return 0;
}