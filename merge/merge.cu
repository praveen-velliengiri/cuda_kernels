#include<cuda_runtime.h>
#include<iostream>
#include<cstdlib>
#include<ctime>
#include<algorithm>
#include <iomanip> // For std::setw
using namespace std;

//each block can process 128 * 8 segment of the output
#define BLK_SIZE 128
#define THR_SIZE 8
#define MAX 1e9+7

__device__ int coRank(int k, int *a, int n, int *b, int m) {
  int low = max(0, k-m), high = min(n, k);
  while (low <= high) {
    int i = low + (high-low)/2;
    int j = k - i;
    if (i > 0 && j < m && a[i-1] > b[j]) {
      high = i-1;
    } else if (j > 0 && i < n && b[j-1] > a[i]) {
      low =  i+1;
    } else {
      return i;
    }
  }
  return -1;
}

__global__ void merge_kernel(int *a, int n, int *b, int m, int *out, int len) {
  unsigned int tid = threadIdx.x, bid = blockIdx.x;
  unsigned int k = (BLK_SIZE * THR_SIZE) * bid + tid * THR_SIZE;

  if (k >= len)
    return;

  int i     = coRank(k, a, n, b, m);
  //printf("corank %d", i);
  if (i == -1) {
    printf("\n coRank -1 for k : %d, n : %d, m : %d", k, n, m);
    return;
  }
  int j     = k - i;

  //since each thread is processing THR_SIZE atmost
  //A[i...i + THR_SIZE] and B[j...j + THR_SIZE] is
  //possible.
  for (int idx = k; idx < (k+THR_SIZE) && idx < len; idx++) {
    int a_ele = i < n ? a[i] : MAX;
    int b_ele = j < m ? b[j] : MAX;
    if (a_ele <= b_ele) {
      out[idx] = a_ele;
      i++;
    } else {
      out[idx] = b_ele;
      j++;
    }
  }
  return;
}

void merge(int *a, int n, int *b, int m, int *out, int len) {
  int k = 0, i = 0, j = 0;
  while (i < n && j < m) {
    if (a[i] <= b[j]) {
      out[k++] = a[i++];
    } else {
      out[k++] = b[j++];
    }
  }
  while (i < n)
    out[k++] = a[i++];
  while (j < m)
    out[k++] = b[j++];
}

void pretty_print(int *gpu, int *cpu, int len) {
    std::cout << std::setw(10) << "CPU" << "  ?  " << std::setw(10) << "GPU" << "\n";
    std::cout << std::string(25, '-') << "\n"; // Separator line

    for (int i = 0; i < len; i++) {
        std::cout << std::setw(10) << cpu[i] << "  ?  " << std::setw(10) << gpu[i] << "\n";
    }
}

void verify(int *gpu, int *cpu, int len) {
  for (int i=0; i<len; i++) {
    if (gpu[i] != cpu[i]) {
      cout << "\n failed at indexes : " << i;
      pretty_print(gpu, cpu, len);
      exit(-1);
    }
  }
  cout << "passed \n";
}

int main(int argc, char **argv) {
  int *a, *b;
  int N = atoi(argv[1]);
  int M = atoi(argv[2]);
  
  a = (int *)malloc(sizeof(int) * N);
  b = (int *)malloc(sizeof(int) * M);

  srand(time(NULL));

  for (int i=0; i<N; i++) {
    a[i] = rand() % 1000;
    //cout << a[i] << " ";
  }
  for (int j=0; j<M; j++) {
    b[j] = rand() % 2000;
    //cout << b[j] << " ";
  }
  sort(a, a+N);
  sort(b, b+M);

  int *da, *db;
  cudaMalloc((void **)&da, sizeof(int) * N);
  cudaMalloc((void **)&db, sizeof(int) * M);

  cudaMemcpy(da, a, sizeof(int) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(db, b, sizeof(int) * M, cudaMemcpyHostToDevice);

  int length = N+M;
  int ngrids = (length + THR_SIZE * BLK_SIZE - 1) / (THR_SIZE * BLK_SIZE);

  int *out;
  cudaMalloc((void **)&out, sizeof(int) * length);

  merge_kernel<<<ngrids, BLK_SIZE>>>(da, N, db, M, out, length);
  cudaDeviceSynchronize();

  int *merged_array = (int *)malloc(sizeof(int) * length);
  cudaMemcpy(merged_array, out, sizeof(int)*length, cudaMemcpyDeviceToHost);

  int *cpu_out = (int *)malloc(sizeof(int) * length);
  merge(a, N, b, M, cpu_out, length);
  verify(merged_array, cpu_out, length);
  return 0;
}