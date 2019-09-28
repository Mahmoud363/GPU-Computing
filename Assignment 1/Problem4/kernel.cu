#include "cuda_runtime.h"

#include "device_launch_parameters.h"
#include<math.h>
#include<time.h>
#include<stdlib.h>
#include <stdio.h>


__global__ void additionMatricesKernel(int *d_a, int *d_b, int *d_c,int size) {


	int i = threadIdx.x + blockIdx.x*blockDim.x;
	for (int j = 0; j < size; j++)
	{
		if (i < size && j < size)
		{
			d_c[i*size + j] = d_a[i*size + j] + d_b[i*size + j];

		}
	}
}


int main() {
	srand(time(0));
	int**h_a, **h_b, **h_c, *d_a, *d_b,*d_c;

	int size;
	printf("Enter the size of your matrix: ");
	scanf("%d", &size);


	h_a = (int **)malloc(size * sizeof(int *));
	h_a[0] = (int *)malloc(size*size * sizeof(int));
	for (int i = 1; i < size; ++i) h_a[i] = h_a[i - 1] + size;

	h_b = (int **)malloc(size * sizeof(int *));
	h_b[0] = (int *)malloc(size*size * sizeof(int));
	for (int i = 1; i < size; ++i) h_b[i] = h_b[i - 1] + size;

	h_c = (int **)malloc(size * sizeof(int *));
	h_c[0] = (int *)malloc(size*size * sizeof(int));
	for (int i = 1; i < size; ++i) h_c[i] = h_c[i - 1] + size;

	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			h_a[i][j] = rand();
			h_b[i][j] = rand();
		}
	}
	int ARRAY_BYTES = size * size * sizeof(int);
	cudaMalloc((void**)&d_a, ARRAY_BYTES);
	cudaMalloc((void**)&d_b, ARRAY_BYTES);
	cudaMalloc((void**)&d_c, ARRAY_BYTES);

	cudaMemcpy(d_a, h_a[0], ARRAY_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b[0], ARRAY_BYTES, cudaMemcpyHostToDevice);

	dim3 grids(ceil(size / 128.0));
	dim3 threads(128);

	additionMatricesKernel << <grids, threads >> > (d_a, d_b,d_c,size);

	cudaMemcpy(h_c[0], d_c, ARRAY_BYTES, cudaMemcpyDeviceToHost);


	// free GPU memory
	cudaFree(d_a);
	cudaFree(d_b);

	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			printf("%d ", h_c[i][j]);
		}
		printf("\n");
	}
	return 0;

}
