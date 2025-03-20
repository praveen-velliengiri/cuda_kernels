//implement a simple reduction(sum) kernel
#include <iostream>
#include <cuda_runtime.h>

// Define constants
//#define N 9999
#define BLK_SIZE 128

using namespace std;
//each thread block is responsible for BLK_SIZE x 2 chunk of input data.
__global__ void reduction_divergence(int *input, int *output, int len) {
    unsigned int segmentStart = blockIdx.x * BLK_SIZE * 2;
    //compute data index for each thread
    unsigned int tid  = threadIdx.x;
    unsigned int iidx = segmentStart + threadIdx.x * 2;

    for (int stride = 1; stride <= blockDim.x; stride *= 2) {
      if (tid % stride == 0) {
        if (iidx+stride < len) {
          input[iidx] += input[iidx+stride];
        }
      }
      __syncthreads();
    }
    if (tid == 0) {
      output[blockIdx.x] = input[iidx];
    }
    return;
}

int cpu_reduction_func(int *input, int len) {
  int sum = 0.0f;
  for (int i=0; i<len; i++) {
    sum += input[i];
  }
  return sum;
}

int main(int argc, char **argv) {
    int N = atoi(argv[1]);
    //cout << N << endl;
    // Host and device pointers
    int *h_A, *h_C;
    int *d_A, *d_C;

    // Allocate memory on host
    h_A = (int*)malloc(N * sizeof(int));

    // Initialize input data
    for (int i = 0; i < N; i++) {
        h_A[i] = static_cast<int>(i);
    }

    // Allocate memory on device
    cudaMalloc((void**)&d_A, N * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, N * sizeof(int), cudaMemcpyHostToDevice);

    // <<< Launch kernel here when implemented >>>
    size_t ngrids = (N+2*BLK_SIZE-1)/(2*BLK_SIZE), nblocks = BLK_SIZE;

    h_C = (int *)malloc(ngrids * sizeof(int));
    cudaMalloc((void**)&d_C, ngrids * sizeof(int));

    cudaEvent_t start, stop;
    float milliseconds = 0;
    
    // Create CUDA events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Record the start event
    cudaEventRecord(start);
    
    // Launch the kernel
    reduction_divergence<<<ngrids, nblocks>>>(d_A, d_C, N);
    cudaDeviceSynchronize();

    // Record the stop event
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Kernel Execution Time: " << milliseconds << " ms" << std::endl;

    // Copy results back to host
    cudaMemcpy(h_C, d_C, ngrids * sizeof(int), cudaMemcpyDeviceToHost);

    int result = cpu_reduction_func(h_A, N);
    int gpu_result = cpu_reduction_func(h_C, ngrids);
    if (abs(result - gpu_result) > 1e-3) {
      std::cout << "\n Actual : " << gpu_result << " != " << " Expected : " << result << "\n";
    }

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_C);
    free(h_A);
    free(h_C);
    return 0;
}