#include<cuda_runtime.h>
#include<iostream>
#define BLK_SIZE 128

__global__ void scan_buffer(int *input, int *output, int N) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * BLK_SIZE + tid;
  
  __shared__ int smem[2][BLK_SIZE];
  int* ptr1 = smem[0], *ptr2 = smem[1];

  //for exclusive scan: load with a right shift by 1.
  ptr1[tid] = (idx-1 < N && idx-1 >= 0)? input[idx-1] : 0;
  ptr2[tid] = 0;

  __syncthreads();
  //int i = 0;
  int *temp;
  //O(log N) steps and O(N log N) additions this is not
  //work efficient than sequential version
  for (int stride = 1; stride <= BLK_SIZE/2; stride <<= 1) {
    //read in idx and write to idx+1 % 2
    if (tid >= stride) {
      //smem[(i+1)%2][tid] = smem[i][tid] + smem[i][tid-stride];
      ptr2[tid] = ptr1[tid] + ptr1[tid-stride];
    } else {
      ptr2[tid] = ptr1[tid];
    }
    __syncthreads();
    temp = ptr1;
    ptr1 = ptr2;
    ptr2 = temp;
    //i = (i + 1) % 2;
  }
  if (idx < N) {
    output[idx] = ptr1[tid];
  }
  return;
}

bool checkResults(int *output, int *input, int N) {
  for (int i=0; i<N; i++) {
    std::cout << input[i] << " -> " << output[i] << "\n";
  }
  /*
  for (int i=0; i<N; i++) {
    if (input[i] != output[i]) {
      std::cout << "failed at idx : " << i << " expect " << input[i] << " recv " << output[i];
      return false;
    } else {
      std::cout << output[i] << " " << input[i] << "\n";
    }
  }*/
  return true;
}

int main(int argc, char **argv) {
  if (!(argc > 1)) {
    std::cout << "expects atleast 2 args";
    return 0;
  }

  int N = atoi(argv[1]);
  int data_size = N * sizeof(int);
  
  int *host_input = (int *)malloc(data_size);
  for (int i=0; i<N; i++)
    host_input[i] = 1;
  
  int *dev_input;
  cudaMalloc((void **)&dev_input, data_size);
  cudaMemcpy(dev_input, host_input, data_size, cudaMemcpyHostToDevice);

  int grids = (N + BLK_SIZE-1) / BLK_SIZE;
  int output_size = data_size;
  int *host_output = (int *)malloc(output_size);
  int *dev_output;
  cudaMalloc((void **)&dev_output, output_size);

  scan_buffer<<<grids, BLK_SIZE>>>(dev_input, dev_output, N);
  cudaDeviceSynchronize();

  cudaMemcpy(host_output, dev_output, output_size, cudaMemcpyDeviceToHost);

  checkResults(host_output, host_input, N);

  cudaFree(dev_input);
  cudaFree(dev_output);

  free(host_output);
  free(host_input);
  return 0;
}