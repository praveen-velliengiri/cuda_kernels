//implement a simple reduction(sum) kernel
#include <iostream>
#include <cuda_runtime.h>

// Define constants
//#define N 9999
#define BLK_SIZE 4

using namespace std;
//each thread block is responsible for BLK_SIZE chunk of input data.
__global__ void scan(int *input, int *output, int *partialsums, int N) {
  unsigned int tid = threadIdx.x;
  unsigned int blockStart = blockIdx.x * blockDim.x;

  __shared__ int smem[BLK_SIZE];
  smem[tid] = input[blockStart + tid];
  __syncthreads();

  for (int stride = 1; stride <= BLK_SIZE/2; stride <<= 1) {
    int reg = 0;
    if (tid >= stride) {
      reg = smem[tid - stride];
    }
    __syncthreads();
    if (tid >= stride) {
      smem[tid] += reg;
    }
    __syncthreads();
  }
  output[blockStart + tid] = smem[tid];
  if (tid == blockDim.x - 1) {
    partialsums[blockIdx.x] = smem[tid];
  }
}

void print(int *input, int n) {
  for (int i=0; i<n; i++)
    cout << input[i] << " ";
}

int main(int argc, char **argv) {
    int N = atoi(argv[1]);
    //cout << N << endl;
    // Host and device pointers
    int *h_A, *h_C;
    int *input, *output, *partialsums;

    // Allocate memory on host
    h_A = (int*)malloc(N * sizeof(int));

    // Initialize input data
    for (int i = 0; i < N; i++) {
        h_A[i] = 1;
    }

    // Allocate memory on device
    cudaMalloc((void**)&input, N * sizeof(int));
    cudaMalloc((void**)&output, N * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(input, h_A, N * sizeof(int), cudaMemcpyHostToDevice);

    // <<< Launch kernel here when implemented >>>
    size_t nblocks = (N+BLK_SIZE-1)/(BLK_SIZE);
    cudaMalloc((void**)&partialsums, nblocks * sizeof(int));

    scan<<<nblocks, BLK_SIZE>>>(input, output, partialsums, N);
    cudaDeviceSynchronize();

    int *host_output = (int *)malloc(N * sizeof(int));
    // Copy results back to host
    cudaMemcpy(host_output, output, N * sizeof(int), cudaMemcpyDeviceToHost);
    
    int *host_partials = (int *)malloc(nblocks * sizeof(int));
    cudaMemcpy(host_partials, partialsums, nblocks * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    print(host_output, N);
    cout << endl;
    print(host_partials, nblocks);

    // Cleanup
    cudaFree(input);
    cudaFree(output);
    cudaFree(partialsums);

    free(h_A);
    free(host_output);
    free(host_partials);
    return 0;
}