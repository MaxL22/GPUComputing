#include <cuda_runtime.h>
#include <iostream>
#define CHECK(call)                                                            \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error in " << __FILE__ << "@" << __LINE__ << ": "     \
                << cudaGetErrorString(err) << std::endl;                       \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)
#define BS 16
#define NB 4

__global__ void streamMQDB(int *A, int *B, int *R) {

  int tid = threadIdx.x + threadIdx.y * blockDim.x;

  // Carico in smem
  __shared__ int As[BS][BS];
  __shared__ int Bs[BS][BS];

  As[threadIdx.y][threadIdx.x] = A[tid];
  Bs[threadIdx.y][threadIdx.x] = B[tid];
  __syncthreads();

  float sum = 0.0f;
  for (int k = 0; k < BS; ++k) {
    sum += As[k][threadIdx.y] * Bs[threadIdx.x][k];
  }
  R[tid] = sum;
}

int main(void) {

  const size_t blockBytes = BS * BS * sizeof(float);
  const size_t totalBytes = NB * blockBytes;

  int *h_A = new int[NB * BS * BS];
  int *h_B = new int[NB * BS * BS];
  int *h_R = new int[NB * BS * BS];

  int *d_A, *d_B, *d_R;
  CHECK(cudaMalloc(&d_A, totalBytes));
  CHECK(cudaMalloc(&d_B, totalBytes));
  CHECK(cudaMalloc(&d_R, totalBytes));

  dim3 grid(1, 1);
  dim3 block(BS, BS);

  cudaStream_t *streams = new cudaStream_t[NB];
  for (int i = 0; i < NB; ++i) {
    CHECK(cudaStreamCreate(&streams[i]));
  }

  for (int i = 0; i < NB; i++) {

    int off = i * BS * BS;

    CHECK(cudaMemcpyAsync(d_A + off, h_A + off, blockBytes,
                          cudaMemcpyHostToDevice, streams[i]));
    CHECK(cudaMemcpyAsync(d_A + off, h_A + off, blockBytes,
                          cudaMemcpyHostToDevice, streams[i]));
    CHECK(cudaMemcpyAsync(d_A + off, h_A + off, blockBytes,
                          cudaMemcpyHostToDevice, streams[i]));

    // lancio kernel sullo stream i
    streamMQDB<<<grid, block, 0, streams[i]>>>(d_A + off, d_B + off, d_R + off);
    // controllo errori di launch
    CHECK(cudaGetLastError());

    // copia asincrona device→host di D_i
    CHECK(cudaMemcpyAsync(h_R + off, d_R + off, blockBytes,
                          cudaMemcpyDeviceToHost, streams[i]));
  }

  return 0;
}
