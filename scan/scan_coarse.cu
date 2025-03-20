#include<cuda_runtime.h>
#include<iostream>
using namespace std;

#define BLK_SIZE 128
#define COARSE_FACTOR 8


//each block is loading blk_dim * coarse_factor data.
__global__ void scan_coarse(int *in, int *out, int N, int *scan_value, int *flags, int *dyn_blk_cnt) {
  __shared__ unsigned int bid;
  unsigned int tid     = threadIdx.x;
  
  if (tid == 0) {
    bid = atomicAdd(dyn_blk_cnt, 1);
  }
  __syncthreads();
  unsigned int bid_s = bid;

  //save this atomic operation by moving to host-code.
  if (tid == 0 && bid_s == 0) {
    atomicAdd(&flags[bid_s], 1);
  }

  unsigned int segment = bid_s * blockDim.x * COARSE_FACTOR;
  //each thread is responsible for loading 8 elements from the global mem to smem
  //instead of loading 8 consecutive elements we have to coalease memory access to gmem
  __shared__ int smem_buffer[BLK_SIZE * COARSE_FACTOR];

  for (int i=0; i<COARSE_FACTOR; i++) {
    unsigned int idx = i * BLK_SIZE + segment + tid;
    smem_buffer[i * BLK_SIZE + tid] = idx < N ? in[idx] : 0;
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
  __shared__ int prevblocksum;

  if (tid == 0) {
    int curr_sum = smem_buffer[(blockDim.x - 1) * COARSE_FACTOR + COARSE_FACTOR-1];
    while(atomicAdd(&flags[bid_s], 0) == 0)  {}
    prevblocksum      = scan_value[bid_s];
    scan_value[bid_s+1] = prevblocksum + curr_sum;
    __threadfence();
    atomicAdd(&flags[bid_s+1], 1);
  }
  __syncthreads();

  for (int i=0; i<COARSE_FACTOR; i++)
    smem_buffer[tid * COARSE_FACTOR + i] += prevblocksum;

  __syncthreads();
  //copy back.
  for (int i=0; i<COARSE_FACTOR; i++) {
    unsigned int idx = i * BLK_SIZE + segment + tid;
    //if (idx < N)
      //out[idx] = smem_buffer[tid * COARSE_FACTOR + i];
    if (idx < N)
      out[idx] = smem_buffer[i * BLK_SIZE + tid];
  }
}

bool checkResult(int *gpu, int *cpu_in, int N) {
  if (gpu[0] != cpu_in[0]) {
    cout << "zeroth mismatch" << endl;
    exit(-1);
  }

  for (int i=1; i<N; i++) {
    cpu_in[i] += cpu_in[i-1];
    if (cpu_in[i] != gpu[i]) {
      cout << "\n mismatch at index " << i << " expect " << cpu_in[i] << " received " << gpu[i];
      exit(-1);
    }
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
  int syn_size = nblocks * sizeof(int);

  int *flags, *scan_value, *dyn_blk_cnt;
  cudaMalloc((void **)&flags, syn_size);
  cudaMalloc((void **)&scan_value, syn_size);
  cudaMalloc((void **)&dyn_blk_cnt, 4);

  cudaMemset(scan_value, 0, syn_size);
  cudaMemset(flags, 0, syn_size);
  cudaMemset(dyn_blk_cnt, 0, 4);

  scan_coarse<<<nblocks, BLK_SIZE>>>(dev_in, dev_out, N, scan_value, flags, dyn_blk_cnt);
  cudaDeviceSynchronize();
  
  out_arr = (int *)malloc(size);
  cudaMemcpy(out_arr, dev_out, size, cudaMemcpyDeviceToHost);

  if (checkResult(out_arr, in_arr, N)) {
    cout << "\n scan passed!";
  }
  free(in_arr);
  free(out_arr);

  cudaFree(scan_value);
  cudaFree(flags);
  cudaFree(dyn_blk_cnt);
  cudaFree(dev_in);
  cudaFree(dev_out);

  return 0;
}