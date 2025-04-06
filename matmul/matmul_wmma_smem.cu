#include<cuda_runtime.h>
#include<mma.h>

#include<iostream>
#include<random>
#include<utility>

#define N 32
#define M 32
#define K 32

using namespace std;
using namespace nvcuda;

template<typename T>
struct Matrix {
  Matrix(int rows, int cols) {
    this->rows = rows;
    this->cols = cols;
    ptr = (T *)malloc(this->rows * this->cols * sizeof(T));
  }
  ~Matrix() {
    if (ptr)
      free(ptr);
  }
  void fill() {
    std::default_random_engine random_engine(0);
    uniform_int_distribution<T> uniform_dist(-128, 127);
    for (int i=0; i<(rows * cols); i++)
      ptr[i] = uniform_dist(random_engine);
  }
  size_t getSize() {
    return rows * cols * sizeof(T);
  }
  static Matrix createMatrix(int rows, int cols) {
    return Matrix(rows, cols);
  }

  T *ptr;
  size_t rows, cols;
};


template<typename T1, typename T2>
__global__ void mm(T1 *a, T1 *b, T2 *c, int m, int k, int n) {

  const int num_wraps_x = 4, num_wraps_y = 4;
  const int wrap_m = 16, wrap_n = 16, wrap_k = 16;

  __shared__ T1 matA[num_wraps_y * wrap_m][256];
  __shared__ T1 matB[256][num_wraps_x * wrap_n];

  //do coaleased memory access.
  //start row in A of this block.
  int blk_row_a = (num_wraps_y * wrap_m) * blockIdx.y;
  int cr_size = blockDim.y, cc_size = blockDim.x;

  for (int chunk_row=0; chunk_row < (blockDim.y * wrap_m); chunk_row += cr_size) {
    int sm_row = chunk_row + threadIdx.y;
    int gm_row = blk_row_a + sm_row;

    //for matrix A this block will do from blk_row_a to blk_row_a + (num_wrap_y * wrap_m)
    //this means k is divisible by blockDim.x
    for (int chunk_col=0; chunk_col < k; chunk_col += cc_size) {
      int sm_col = chunk_col + threadIdx.x;

      int a_linear_addr = gm_row * k + sm_col;
      matA[sm_row][sm_col] = a[a_linear_addr];
    }
  }

  //start col in B of this block.
  //(4x128) threads here.
  int blk_col_b = (num_wraps_x * wrap_n) * blockIdx.x;
  cc_size = num_wraps_x, cr_size = blockDim.x;
  for (int chunk_col=0; chunk_col < (num_wraps_x * wrap_n); chunk_col += cc_size) {//maps to thread y dim
    int sm_col = chunk_col + threadIdx.y;
    int gm_col = blk_col_b + sm_col;

    for (int chunk_row=0; chunk_row < k; chunk_row += cr_size) {
      int sm_row = chunk_row + threadIdx.x;

      int b_linear_addr = sm_row * n + gm_col;
      matB[sm_row][sm_col] = b[b_linear_addr]; 
    }
  }

  __syncthreads();
  //mat(64, k) and mat(k, 64)
  const int wrap_size = 32;

  int lda = 256, ldb = num_wraps_x * wrap_n;
  int local_wrap_x = threadIdx.x / wrap_size;
  int local_wrap_y = threadIdx.y;

  //each wrap is responsible for 16 x 16 chunk for output matrix.
  //a_frag
  wmma::fragment<wmma::matrix_a, wrap_m, wrap_n, wrap_k, T1, wmma::row_major> a_frag;
  //b_frag
  wmma::fragment<wmma::matrix_b, wrap_m, wrap_n, wrap_k, T1, wmma::row_major> b_frag;
  //accum_frag
  wmma::fragment<wmma::accumulator, wrap_m, wrap_n, wrap_k, T2> accum_frag;

  wmma::fill_fragment(accum_frag, 0);

  for (int i=0; i<k; i+=wrap_k) {
    //a smem matrix.
    int a_wrap_row = local_wrap_y * wrap_m, a_wrap_col = i;
    //b smem matrix.
    int b_wrap_row = i, b_wrap_col = local_wrap_x * wrap_n;

    T1 *a_sm_ptr = &matA[a_wrap_row][a_wrap_col];
    T1 *b_sm_ptr = &matB[b_wrap_row][b_wrap_col];

    //load
    wmma::load_matrix_sync(a_frag, a_sm_ptr, lda);
    wmma::load_matrix_sync(b_frag, b_sm_ptr, ldb);

    wmma::mma_sync(accum_frag, a_frag, b_frag, accum_frag);
  }

  //store
  int wrap_x = (blockDim.x * blockIdx.x + threadIdx.x) / wrap_size;
  int wrap_y = (blockDim.y * blockIdx.y) + threadIdx.y;

  int c_start_row = wrap_y * 16, c_start_col = wrap_x * 16;
  T2 *c_ptr = c + (c_start_row * n + c_start_col);
  wmma::store_matrix_sync(c_ptr, accum_frag, n, wmma::mem_row_major);
  return;
}

template<typename T1, typename T2>
Matrix<T2> launch_matmul_kernel(Matrix<T1>& A, Matrix<T1>& B) {
  //device matrix allocate
  T1 *dmatrix_a, *dmatrix_b;
  T2 *dmatrix_c;

  Matrix<T2> C = Matrix<T2>::createMatrix(A.rows, B.cols);

  cudaMalloc((void **)&dmatrix_a, A.getSize());
  cudaMalloc((void **)&dmatrix_b, B.getSize());
  cudaMalloc((void **)&dmatrix_c, C.getSize());

  //device matrix copy
  cudaMemcpy(dmatrix_a, A.ptr, A.getSize(), cudaMemcpyHostToDevice);
  cudaMemcpy(dmatrix_b, B.ptr, B.getSize(), cudaMemcpyHostToDevice);

  //kernel stepup
  //each block has (4 x 4 wraps)
  const int num_wraps_x = 4;//horizontal
  const int num_wraps_y = 4;//vertical
  const int wrap_m = 16;
  const int wrap_n = 16;
  int n = C.cols, m = C.rows;
  dim3 block, grid;
  block.x = num_wraps_x * 32;
  block.y = num_wraps_y;

  //output wil be of size (m * n)
  //map horizontal blocks to columns
  //vertical blocks to rows
  grid.x  = (n + num_wraps_x * wrap_n - 1) / (num_wraps_x * wrap_n);
  grid.y  = (m + num_wraps_y * wrap_m - 1) / (num_wraps_y * wrap_m);

  std::cout << "\n grid  dim : " << grid.x <<  "," << grid.y;
  std::cout << "\n block dim : " << block.x << "," << block.y;
  
  cudaFuncSetAttribute(mm<int8_t, int>, 
    cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);

  mm<int8_t, int><<<grid, block>>>(dmatrix_a, dmatrix_b, dmatrix_c, A.rows, A.cols, B.cols);
  cudaDeviceSynchronize();
  //copy result
  cudaMemcpy(C.ptr, dmatrix_c, C.getSize(), cudaMemcpyDeviceToHost);
  return C;
}

template<typename T1, typename T2>
Matrix<T2> matrixMultiply(Matrix<T1> &A, Matrix<T1> &B) {
  Matrix<T2> C = Matrix<T2>::createMatrix(A.rows, B.cols);
  
  int m = A.rows, k = A.cols, n = B.cols;
  for (int i=0; i<m; i++) {
    for (int j=0; j<n; j++) {
      T2 sum = 0;
      for (int z=0; z<k; z++) {
        sum += A.ptr[i * k + z] * B.ptr[z * n + j];
      }
      C.ptr[i * n + j] = sum;
    }
  }
  return C;
}

template<typename T>
bool verify(Matrix<T> &gpu_c, Matrix<T> &cpu_c) {
  if ((gpu_c.rows != cpu_c.rows) || (gpu_c.cols != cpu_c.cols)) {
    std::cout << "\n shape mismatch";
    return false;
  }

  for (int i=0; i<cpu_c.rows; i++) {
    for (int j=0; j<cpu_c.cols; j++) {
      if (gpu_c.ptr[i * cpu_c.cols + j] != cpu_c.ptr[i * cpu_c.cols + j])
        return false;
    }
  }
  return true;
}

//Matrix<int8_t>(rows, cols);
int main(int argc, char** argv) {
  if (argc < 4) {
    return 0;
  }
  int m = atoi(argv[1]), k = atoi(argv[2]), n = atoi(argv[3]);

  using T1 = int8_t;
  using T2 = int;
  //input allocate
  Matrix A = Matrix<T1>::createMatrix(m, k);
  Matrix B = Matrix<T1>::createMatrix(k, n);
  
  //fill input matrices
  A.fill();
  B.fill();

  auto mma_c = launch_matmul_kernel<T1, T2>(A, B);

  auto cpu_c = matrixMultiply<T1, T2>(A, B);

  if(!verify(mma_c, cpu_c)) {
    std::cout << "\n mismatch failed";
  } else {
    std::cout << "\n kernel passed";
  }
  return 0;
}