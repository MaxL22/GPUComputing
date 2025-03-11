#include <cstdio>
#include <stdio.h>

//__global__ dice che è chiamata dalla CPU ma eseguita dalla GPU
__global__ void helloFromGPU (void) {
  // Built in, serve per l'indicizzazione dei thread
  // tID è il thread
  int tID = threadIdx.x;
  printf("Hello world from GPU (thread %d) \n", tID);
}

int main(void){

  // Triple parentesi indicano i thread
  printf("Testing \n");
  
  cudaSetDevice(0);

    
  helloFromGPU<<<1,64>>>();
  cudaDeviceSynchronize();

  
    
  return 0;
}

