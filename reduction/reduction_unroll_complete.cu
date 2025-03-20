//implement a simple reduction(sum) kernel
#include <iostream>
#include <cuda_runtime.h>

// Define constants
//#define N 9999
#define BLK_SIZE 128
#define COARSE 4

using namespace std;
//usaully we have to increase the rate at which our kernel gets the data.
//use mechanisms like memory coalescing and conflict-free access at first.


//each thread block is responsible for BLK_SIZE x COARSE chunk of input data.
__global__ void reduction_divergence(int *input, int *output, int len) {
    unsigned int segmentStart = blockIdx.x * BLK_SIZE * COARSE;
    //compute data index for each thread
    unsigned int tid  = threadIdx.x;

    __shared__ int smem[BLK_SIZE];
    smem[tid] = 0;
    __syncthreads();

    //reduce the smem size of blocksize.
    //load 128 consecutive bytes and so on.
    //check if it is 4-sector returns per loop. //how to check?
    for (int i = 0; i<COARSE; i++) {
      int iidx   = segmentStart + (BLK_SIZE * i) + tid;
      if (iidx < len)
        smem[tid] += input[iidx];
    }
    __syncthreads();

    //since we have smem of block_size, start with block_size / 2 stride
    if (tid < 64) smem[tid] += smem[tid + 64];
    __syncthreads();

    //wrap reduce unroll
    //volatile required to prevent re-ordering by compiler.
    volatile int *vsem = smem;
    if (tid < 32) vsem[tid] += vsem[tid + 32];
    __syncthreads();

    if (tid < 16) vsem[tid] += vsem[tid + 16];
    if (tid < 8)  vsem[tid] += vsem[tid + 8];
    if (tid < 4)  vsem[tid] += vsem[tid + 4];
    if (tid < 2)  vsem[tid] += vsem[tid + 2];
    if (tid < 1)  vsem[tid] += vsem[tid + 1];
  
    if (tid == 0) {
      output[blockIdx.x] = vsem[tid];
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
    size_t ngrids = (N+COARSE*BLK_SIZE-1)/(COARSE*BLK_SIZE), nblocks = BLK_SIZE;

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
      std::exit(-1);
    }

    std::cout << "\n reduction complete \n";
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_C);
    free(h_A);
    free(h_C);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}