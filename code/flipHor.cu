// nvcc -arch=sm_86 flipHor.cu -I utils/PPM utils/PPM/ppm.cpp

#include "utils/PPM/ppm.h"
#include "utils/common.h"
#include <cstddef>
#include <cstdio>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 1024

// TODO It says it's wrong, but idk y tho

/*
 * Kernel function to flip an image horizontally
 */

/*
 * Kernel 1D that flips inplace the PPM image horizontally:
 * each thread only flips a single pixel (R,G,B)
 */
__global__ void ppm_flipH_GPU(color *input, color *output, size_t width,
                              size_t height) {

  // Total number of pixels and linearized thread Id
  const int nump = width * height;
  int idt = blockIdx.x * blockDim.x + threadIdx.x;

  int numrow = (int)idt / width;
  int numcol = idt % (width - 1);

  if (idt < nump) {
    for (int i = 0; i < 3; i++) {
      *(output + (idt * 3) + i) =
          *(input + ((numrow * width + (width - numcol)) * 3) + i);
    }
  }
}

/*
 * MAIN
 */
int main(int argc, char **argv) {

  // PPM images
  PPM *ppm, *ppm1, *ppm2; // Where images are stored in CPU

  // load a PPM image from file
  char path[] = "images/dog.ppm";
  ppm = ppm_load(path);
  ppm1 = ppm_copy(ppm);
  uint WIDTH = ppm->width;
  uint HEIGHT = ppm->height;
  printf("PPM image size (w x h): %d x %d\n", WIDTH, HEIGHT);

  // set main params
  size_t nBytes = WIDTH * HEIGHT * sizeof(pel);

  // Allocate GPU buffer for the input and output images
  color *input_dev, *output_dev;
  CHECK(cudaMalloc(&input_dev, nBytes));
  CHECK(cudaMalloc(&output_dev, nBytes));

  // copy image from CPU to GPU
  cudaMemcpy(input_dev, ppm1->image, nBytes, cudaMemcpyHostToDevice);

  // invoke kernels (define grid and block sizes)
  dim3 block(BLOCK_SIZE, 1, 1);
  int numgrids = (WIDTH * HEIGHT) / BLOCK_SIZE + 1;
  dim3 grid((numgrids), 1, 1);

  double startGPU = seconds();

  ppm_flipH_GPU<<<grid, block>>>(input_dev, output_dev, WIDTH, HEIGHT);
  // ppm_flipH_GPU<<<1, 1>>>(input_dev, output_dev, WIDTH, HEIGHT);

  double stopGPU = seconds() - startGPU;

  // copy image from GPU to CPU
  cudaMemcpy(ppm1->image, output_dev, nBytes, cudaMemcpyDeviceToHost);

  cudaFree(input_dev);
  cudaFree(output_dev);

  // check results with CPU
  ppm2 = ppm_copy(ppm);
  double startCPU = seconds();
  ppm_flipH(ppm2);
  double stopCPU = seconds() - startCPU;

  // double stopCPU = seconds() - start;
  char res = ppm_equal(ppm1, ppm2) ? 'Y' : 'N';
  printf("Are equal? %c\n", res);
  ppm_write(ppm2, "output_flippedV_CPU.ppm");

  // times & speedup
  printf("CPU elapsed time: %.4f (msec) \n", stopCPU * 1000);
  printf("GPU elapsed time: %.4f (msec) - Speedup %.1f\n", stopGPU * 1000,
         stopCPU / stopGPU);

  return (EXIT_SUCCESS);
}
