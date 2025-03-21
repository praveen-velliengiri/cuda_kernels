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

//do coarsening and privatization.
//fill shared memory with special value HIST_SIZE and avoid checking it.
//here we are only using 
__global__ void histogram_opt_uniform(unsigned char *arr, int *out, int n) {
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
  for (int i=0; i<PRIV_SIZE/BLK_SIZE; i++) {
    private_histogram[i * BLK_SIZE + tid ] = 0;
  }

  __syncthreads();

  //do atomic add to private histogram
  for (int i=0; i<CFACTOR; i++) {
    int value = smem[i * BLK_SIZE + tid];
    atomicAdd(&private_histogram[value], 1);
  }
  __syncthreads();


  for (int i=0; i<PRIV_SIZE/BLK_SIZE; i++) {
    int value = private_histogram[i * BLK_SIZE + tid];
    atomicAdd(&out[i * BLK_SIZE + tid], value);
  }
  return;
}