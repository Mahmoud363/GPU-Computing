#include "cuda_runtime.h"

#include "device_launch_parameters.h"
#include<math.h>
#include<time.h>
#include<stdlib.h>
#include <stdio.h>

#define ROWS 520
#define COLS 260

__global__ void multiplyTwoKernel(int *d_img, int *d_res, int row, int col) {


	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	if (i < row && j < col)
	{
		d_res[i*col + j] = d_img[i*col + j] * 2;

	}
}


int main() {
	srand(time(0));
	int**h_img, **h_res, *d_img, *d_res;

	h_img = (int **)malloc(ROWS * sizeof(int *));
	h_img[0] = (int *)malloc(ROWS*COLS * sizeof(int));
	for (int i = 1; i < ROWS; ++i) h_img[i] = h_img[i - 1] + COLS;

	h_res = (int **)malloc(ROWS * sizeof(int *));
	h_res[0] = (int *)malloc(ROWS*COLS * sizeof(int));
	for (int i = 1; i < ROWS; ++i) h_res[i] = h_res[i - 1] + COLS;


	for (int i = 0; i < ROWS; i++)
	{
		for (int j = 0; j < COLS; j++)
		{
			h_img[i][j] = rand() % 256;
		}
	}
	int ARRAY_BYTES = ROWS * COLS * sizeof(int);
	cudaMalloc((void**)&d_img, ARRAY_BYTES);
	cudaMalloc((void**)&d_res, ARRAY_BYTES);

	cudaMemcpy(d_img, h_img[0], ARRAY_BYTES, cudaMemcpyHostToDevice);

	dim3 grids(ceil(ROWS / 16.0), ceil(COLS / 16.0));
	dim3 threads(16, 16);

	multiplyTwoKernel << <grids, threads >> > (d_img, d_res, ROWS, COLS);

	cudaMemcpy(h_res[0], d_res, ARRAY_BYTES, cudaMemcpyDeviceToHost);


	// free GPU memory
	cudaFree(d_img);
	cudaFree(d_res);

	for (int i = 0; i < ROWS; i++)
	{
		for (int j = 0; j < COLS; j++)
		{
			printf("%d ", h_res[i][j]);
		}
		printf("\n");
	}
	return 0;

}
