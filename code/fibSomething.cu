#include <cstdio>
#include <stdio.h>

#define SIZE 5

/*
 * Show DIMs & IDs for grid, block and thread
 * Un thread stampa il proprio indice solo se è pari ad un numero nelle sequenza
 * di Fibonacci
 */
__global__ void checkIndex(void) {

  int idbk = blockIdx.y * gridDim.x + blockIdx.x;
  int s =
      idbk * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;

  int a = 0, b = 1, c = 0;
  int count = 1;
  while (c < s) {
    c = a + b;
    a = b;
    b = c;

    count++;
  }

  if (s == c) {
    printf("Hello, I'm thread num %d, the %d th fib number \n", s, count);
  }
}

int main(int argc, char **argv) {

  // grid and block structure
  dim3 grid(SIZE, SIZE);
  dim3 block(SIZE, SIZE);

  // Print from host
  printf("Print from host:\n");
  printf("grid.x = %d\t grid.y = %d\t grid.z = %d\n", grid.x, grid.y, grid.z);
  printf("block.x = %d\t block.y = %d\t block.z %d\n\n", block.x, block.y,
         block.z);

  // Print from device
  checkIndex<<<grid, block>>>();

  // reset device
  cudaDeviceReset();
  return (0);
}
