#include <stdio.h>

__global__ void checkIndex(void) {
	printf("threadIdx:(%d, %d, %d) blockIdx:(%d, %d, %d) "
					"blockDim:(%d, %d, %d) gridDim:(%d, %d, %d)\n",
					threadIdx.x, threadIdx.y, threadIdx.z,
					blockIdx.x, blockIdx.y, blockIdx.z,
					blockDim.x, blockDim.y, blockDim.z,
					gridDim.x,gridDim.y,gridDim.z);
}

/*
* MAIN
*/
int main(int argc, char **argv) {

	// grid and block definition
	dim3 block(4);
	dim3 grid(3);

	// Print from host
	printf("Print from host:\n");
	printf("grid.x = %d\t grid.y = %d\t grid.z = %d\n", grid.x, grid.y, grid.z);
	printf("block.x = %d\t block.y = %d\t block.z %d\n\n", block.x, block.y, block.z);

	// Print from device
	printf("Print from device:\n");
	checkIndex<<<grid, block>>>();

	// reset device
	cudaDeviceReset();
	return(0);
}
