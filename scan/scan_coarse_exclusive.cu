#include<cuda_runtime.h>
#include<iostream>
using namespace std;

#define BLK_SIZE 128
#define COARSE_FACTOR 8


//each block is loading blk_dim * coarse_factor data.
__global__ void scan_coarse(int *in, int *out, int N) {

  unsigned int segment = blockIdx.x * blockDim.x * COARSE_FACTOR;
  unsigned int tid     = threadIdx.x;
  //each thread is responsible for loading 8 elements from the global mem to smem
  //instead of loading 8 consecutive elements we have to coalease memory access to gmem
  __shared__ int smem_buffer[BLK_SIZE * COARSE_FACTOR];

  for (int i=0; i<COARSE_FACTOR; i++) {
    int idx = i * BLK_SIZE + segment + tid - 1;
    smem_buffer[i * BLK_SIZE + tid] = idx < N && idx >= 0 ? in[idx] : 0;
  }
  
  __syncthreads();
  //tid to tid+7
  for (int i=1; i<COARSE_FACTOR; i++) {
    //tid * COARSE_FACTOR
    smem_buffer[tid * COARSE_FACTOR + i] += smem_buffer[tid * COARSE_FACTOR + i-1];
  }
  __shared__ int scan_buffer_1[BLK_SIZE], scan_buffer_2[BLK_SIZE];
  int *read = scan_buffer_1, *write = scan_buffer_2, *temp;
  read[tid]   = smem_buffer[tid * COARSE_FACTOR + COARSE_FACTOR-1];
  __syncthreads();

  //scan
  for (int stride = 1; stride <= BLK_SIZE/2; stride <<= 1) {
    if (tid >= stride) { //do useful work
      write[tid] = read[tid] + read[tid-stride];
    } else { //copy data
      write[tid] = read[tid];
    }
    __syncthreads();
    //swap
    temp  = read;
    read  = write;
    write = temp;
  }
  
  if (tid > 0) {
    for (int i=0; i<COARSE_FACTOR; i++) {
      smem_buffer[tid * COARSE_FACTOR + i] += read[tid-1];
    }
  }
  __syncthreads();
  //printf("%d : %d\n",tid,smem_buffer[tid]);
  //copy back.
  for (int i=0; i<COARSE_FACTOR; i++) {
      int idx = i * BLK_SIZE + segment + tid;
    //if (idx < N)
      //out[idx] = smem_buffer[tid * COARSE_FACTOR + i];
    if (idx < N)
      out[idx] = smem_buffer[i * BLK_SIZE + tid];
  }
}

bool checkResult(int *gpu, int *cpu_in, int N) {
  int sum = cpu_in[0];

  for (int i=1; i<N; i++) {
    if (sum != gpu[i]) {
      cout << "\n mismatch at index " << i << " expect " << cpu_in[i] << " received " << gpu[i];
      exit(-1);
    }
    sum += cpu_in[i];
  }
  return true;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    cout << "\n requires atleast one argument";
    exit(-1);
  }
  int N = atoi(argv[1]);

  int *in_arr, *out_arr, *dev_in, *dev_out;
  int size = N * sizeof(int);
  in_arr = (int *)malloc(size);

  cudaMalloc((void**)&dev_in, size);
  cudaMalloc((void**)&dev_out, size);

  //memset(in_arr, 1, size);
  for (int i=0; i<N; i++)
    in_arr[i] = 1;

  cudaMemcpy(dev_in, in_arr, size, cudaMemcpyHostToDevice);
  int bsize = BLK_SIZE * COARSE_FACTOR;
  int nblocks = (N + bsize - 1) / bsize;
  //cout << nblocks << " " << BLK_SIZE;
  scan_coarse<<<nblocks, BLK_SIZE>>>(dev_in, dev_out, N);
  cudaDeviceSynchronize();
  out_arr = (int *)malloc(size);
  cudaMemcpy(out_arr, dev_out, size, cudaMemcpyDeviceToHost);

  if (checkResult(out_arr, in_arr, N)) {
    cout << "\n scan ok!";
  }
  free(in_arr);
  free(out_arr);
  cudaFree(dev_in);
  cudaFree(dev_out);

  return 0;
}