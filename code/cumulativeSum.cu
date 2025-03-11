#include <cstdio>
#include <cstdlib>
#include <stdlib.h>
#define SIZE 15
#define NUM 2

const int vec_size = SIZE * sizeof(float);

void __global__ cumulativeSum(int *a, int *b, int size) {
  int idx = threadIdx.x;

  if (idx < size) {
    int p = 0;
    for (int i = 0; i <= idx; i++) {
      p += a[i];
    }
    b[idx] = p;
  }
}

int main() {

  // allocate space for vectors in host memory
  int *values, *sums;

  values = (int *)malloc(vec_size);
  sums = (int *)malloc(vec_size);

  // put 1 in the array
  for (int i = 0; i < SIZE; i++) {
    values[i] = 1;
  }

  printf("Values: \n");
  for (int i = 0; i < SIZE; i++) {
    printf("%d ", values[i]);
  }
  printf("\n");

  // allocate space for vectors in device memory
  int *dev_Values, *dev_Sums;
  cudaMalloc(&dev_Values, vec_size);
  cudaMalloc(&dev_Sums, vec_size);

  // copy vectors A and B from host to device:
  cudaMemcpy(dev_Values, values, vec_size, cudaMemcpyHostToDevice);
  // cudaMemcpy(dev_Sums, sums, vec_size, cudaMemcpyHostToDevice);

  // launch the vector adding kernel
  cudaSetDevice(0);
  cumulativeSum<<<1, vec_size>>>(dev_Values, dev_Sums, vec_size);

  // wait for the kernel to finish execution
  cudaDeviceSynchronize();

  // copy from device memory
  cudaMemcpy(sums, dev_Sums, vec_size, cudaMemcpyDeviceToHost);
  cudaFree(dev_Values);
  cudaFree(dev_Sums);

  // print some results
  printf("Values after sum: \n");
  for (int i = 0; i < SIZE; i++) {
    printf("%d ", sums[i]);
  }

  // free host vecs
  free(values);
  free(sums);

  printf("\n");
}
