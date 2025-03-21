#include<cuda_runtime.h>
#include<iostream>
using namespace std;
#define BLK_SIZE 128

__global__ void histogram(unsigned char *arr, int *out, int n) {
  unsigned int tid = threadIdx.x;
  unsigned int idx = blockIdx.x * blockDim.x + tid;
  if (idx < n) {
    unsigned int value = arr[idx];
    atomicAdd(&out[value], 1);
  }
  return;
}

bool checkAnswer(int *gpu, unsigned char *cpu, int n) {
  int *out = (int *)malloc(sizeof(int) * 256);
  memset(out, 0, sizeof(int) * 256);
  for (int i=0; i<n; i++)
    out[cpu[i]]++;
  
  for (int i=0; i<256; i++) {
    if (out[i] != gpu[i]) {
      cout << "mismatch at idx " << i << " expect " << out[i] << " but received " << gpu[i];
      return false;
    }
  }
  return true;
}

int main(int argc, char **argv) {
  
  if(argc < 2) {
    cout << "usage: requires a size of the array";
    exit(-1);
  }
  int n = atoi(argv[1]);
  size_t size = sizeof(unsigned char) * n;
  
  unsigned char *arr = (unsigned char *)malloc(size);
  for (int i=0; i<n; i++) {
    arr[i] = i % 256;
  }

  unsigned char *dev_arr;
  int *dev_out;
  //input
  cudaMalloc((void **)&dev_arr, size);
  cudaMemcpy(dev_arr, arr, size, cudaMemcpyHostToDevice);
  //output
  cudaMalloc((void **)&dev_out, sizeof(int) * 256);
  cudaMemset(dev_out, 0, sizeof(int) * 256);

  int nblocks = (n + BLK_SIZE - 1) / BLK_SIZE;
  histogram<<<nblocks, BLK_SIZE>>>(dev_arr, dev_out, n);
  cudaDeviceSynchronize();

  int *out = (int *)malloc(sizeof(int) * 256);
  cudaMemcpy(out, dev_out, sizeof(int) * 256, cudaMemcpyDeviceToHost);

  if (checkAnswer(out, arr, n)) {
    cout << "\n histogram pass ";
  } else {
    exit(-1);
  }
  free(arr);
  free(out);
  cudaFree(dev_arr);
  cudaFree(dev_out);
  return 0;
}