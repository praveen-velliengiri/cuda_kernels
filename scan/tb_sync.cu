#include<iostream>
#include<cuda_runtime.h>
using namespace std;
#define BLK_SIZE 32

__global__ void tb_sync(int *out, int *scan_value, int *flags, int *dyn_blk_cnt) {
  int bid_s;

  if (threadIdx.x == 0) {
    bid_s = atomicAdd(dyn_blk_cnt, 1);
  }
  if (threadIdx.x == 0 && bid_s == 0)
      atomicAdd(&flags[bid_s], 1);
  
  __syncthreads();
  int getvalue;
  if (threadIdx.x == 0) {
    while (atomicAdd(&flags[bid_s], 0) == 0) {}
    getvalue = scan_value[bid_s];
    scan_value[bid_s + 1] = getvalue + 1;
    __threadfence();
    atomicAdd(&flags[bid_s + 1], 1);
  }
  __syncthreads();
  if (threadIdx.x == 0)
     out[bid_s] = getvalue;
}


int main(int argc, char **argv) {
  if (argc < 2) {
    cout << "\n requires atleast one argument";
    exit(-1);
  }
  int N = atoi(argv[1]);

  int *in_arr, *out_arr, *dev_in, *dev_out;
  int size = N * sizeof(int);

  cudaMalloc((void**)&dev_out, size);

  //cudaMemcpy(dev_in, in_arr, size, cudaMemcpyHostToDevice);
  //int bsize       = BLK_SIZE * COARSE_FACTOR;
  
  int bsize       = 32;
  int *flags      = (int *)malloc(N * sizeof(int));
  int *scan_value = (int *)malloc(N * sizeof(int));

  //memset(scan_value, 0, nblocks * sizeof(int));
  //memset(flags, 0, nblocks * sizeof(int));
  for (int i=0; i<N; i++) {
    scan_value[i] = 0;
    flags[i] = 0;
  }

  int* dyn_blk_cnt = (int *)malloc(sizeof(int));
  *dyn_blk_cnt = 0;
  //cout << nblocks << " " << BLK_SIZE;
  tb_sync<<<N, BLK_SIZE>>>(dev_out, scan_value, flags, dyn_blk_cnt);
  cudaDeviceSynchronize();
  
  out_arr = (int *)malloc(size);
  cudaMemcpy(out_arr, dev_out, size, cudaMemcpyDeviceToHost);

  cout << out_arr[0] << " " << out_arr[1];

  free(out_arr);
  cudaFree(dev_out);

  return 0;
}