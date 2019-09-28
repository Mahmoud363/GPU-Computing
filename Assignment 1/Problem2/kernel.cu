
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<math.h>
#include<time.h>
#include<stdlib.h>
#include <stdio.h>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b, const int n)
{
    int i =blockIdx.x*blockDim.x+ threadIdx.x;
	if(i*2<n)
		c[i * 2] = a[i * 2] + b[i * 2];
	if((i*2+1)<n)
		c[i * 2 + 1] = a[i * 2 + 1] + b[i * 2 + 1];
}

int main()
{
	int arraySize;
	printf("Enter the size of your array: ");
	scanf("%d", &arraySize);
	int* a = (int*)malloc(arraySize * sizeof(int));

	 // if memory cannot be allocated
	 if (a == NULL)
	 {
		 printf("Error! memory not allocated.");
		 exit(0);
	 }
	 int* b = (int*)malloc(arraySize * sizeof(int));

	 // if memory cannot be allocated
	 if (b == NULL)
	 {
		 printf("Error! memory not allocated.");
		 exit(0);
	 }
	 int* c = (int*)malloc(arraySize * sizeof(int));

	 // if memory cannot be allocated
	 if (c == NULL)
	 {
		 printf("Error! memory not allocated.");
		 exit(0);
	 }
	srand(time(0));
	for (int i = 0; i < arraySize; i++)
	{
		a[i] = rand();
		b[i] = rand();
	}



    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

	for (int i = 0; i < arraySize; i++)
	{
		printf("%d", c[i]);
		((i+1) % 20 == 0) ? printf("\n") : printf(" ");
	}
   

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    free(a);
    free(b);
    free(c);
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	dim3 gridSize(ceil(ceil(size/2.0) / 128.0), 1, 1);
	dim3 blockSize(128, 1, 1);
    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<gridSize, blockSize>>>(dev_c, dev_a, dev_b, size);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
