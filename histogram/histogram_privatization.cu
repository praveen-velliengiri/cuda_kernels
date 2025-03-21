#include<cuda_runtime.h>
#include<utils.hpp>
#include<histogram_kernels.cuh>

using namespace std;
/*
#define BLK_SIZE 128
#define CFACTOR  8
#define HIST_SIZE 255 //0-254 values + 255 boundary value.
#define PRIV_SIZE (HIST_SIZE + 1)
#define WRAP_SIZE 32
//no opt 10M = 0.813ms
//opt 10M    = 1.111ms why optimized is bad?

__global__ void histogram(unsigned char *arr, int *out, int n) {
  unsigned int tid = threadIdx.x;
  unsigned int idx = blockIdx.x * blockDim.x + tid;
  if (idx < n) {
    unsigned int value = arr[idx];
    atomicAdd(&out[value], 1);
  }
  return;
}

//do coarsening and privatization.
//fill shared memory with special value HIST_SIZE and avoid checking it.
//here we are only using 
__global__ void histogram_opt(unsigned char *arr, int *out, int n) {
  unsigned int tid = threadIdx.x;
  unsigned int segment = blockIdx.x * blockDim.x * CFACTOR;
  
  __shared__ unsigned char smem[BLK_SIZE * CFACTOR];
  __shared__ int private_histogram[PRIV_SIZE];

  //if (tid == 0) {printf("%d", HIST_SIZE);}
  for (int i=0; i<CFACTOR; i++) {
    unsigned int idx = segment + i * BLK_SIZE + tid;
    smem[i * BLK_SIZE + tid] = idx < n ? arr[idx] : HIST_SIZE;
  }

  __syncthreads();

  //initialize histogram to zero in one wrap.
  if (tid < WRAP_SIZE) {
    for (int i=0; i<PRIV_SIZE/WRAP_SIZE; i++) {
      private_histogram[i * WRAP_SIZE + tid] = 0;
    }
  }
  __syncthreads();

  //do atomic add to private histogram
  for (int i=0; i<CFACTOR; i++) {
    int value = smem[i * BLK_SIZE + tid];
    atomicAdd(&private_histogram[value], 1);
  }
  __syncthreads();

  if (tid < WRAP_SIZE) {
    for (int i=0; i<PRIV_SIZE/WRAP_SIZE; i++) {
      int value = private_histogram[i * WRAP_SIZE + tid];
      atomicAdd(&out[i * WRAP_SIZE + tid], value);
    }
  }
  return;
}
*/


int main(int argc, char **argv) {
  
  if(argc < 3) {
    cout << "usage: requires a size of the array";
    exit(-1);
  }
  int n = atoi(argv[1]);
  int k = atoi(argv[2]);

  unsigned char *arr = getmem<unsigned char>(n);
  size_t arr_size = sizeof(unsigned char) * n;
  fillran<unsigned char>(arr, n, 255);
  //fillseq<unsigned char>(arr, 10, n);

  unsigned char *darr;
  int *dout;
  size_t out_size = sizeof(int) * HIST_SIZE;
  cudaMalloc((void **)&darr, arr_size);
  cudaMalloc((void **)&dout, out_size);
  cudaMemset(dout, 0, out_size);

  cudaMemcpy(darr, arr, arr_size, cudaMemcpyHostToDevice);

  int nblocks = (n + BLK_SIZE - 1) / BLK_SIZE;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  if (k == 0)
    histogram_opt<<<nblocks, BLK_SIZE>>>(darr, dout, n);
  else if (k == 1)
    histogram<<<nblocks, BLK_SIZE>>>(darr, dout, n);
  else
    histogram_opt_uniform<<<nblocks, BLK_SIZE>>>(darr, dout, n);

  cudaDeviceSynchronize();
  
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float time = 0.0f;
  cudaEventElapsedTime(&time, start, stop);

  int *out = getmem<int>(HIST_SIZE);
  cudaMemcpy(out, dout, out_size, cudaMemcpyDeviceToHost);

  int *cpu = getmem<int>(HIST_SIZE);
  //cpu histogram
  for (int i=0; i<n; i++) {
    ++cpu[arr[i]];
  }

  if (check(cpu, out, HIST_SIZE)) {
    cout << "\nHistogram kernel pass\n";
    printf("Kernel execution time: %.3f ms\n", time);
  }

  return 0;
}