#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<math.h>
#include<time.h>
#include<stdlib.h>
#include <stdio.h>
#include <ctime>
#include <cmath>
#include "helper.h"
using namespace std;

#define BLOCKS 1024



float duration_gpu, duration_cpu, duration_kernel, duration_cpumem;


__global__ void Dot_Product(double* d_A, double* d_B, int len) {
	__shared__ double Shared_mem[BLOCKS*2];

	// each thread loads one element from global to shared mem
	int t = threadIdx.x;
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	//Shared_mem[t] = 0;
	Shared_mem[t] = 0;

	if (i < len)
	{
		Shared_mem[t] = d_A[i] * d_B[i];
	}
	if (i + 1024 < len)
	{
		Shared_mem[t + 1024] = d_A[i + 1024] * d_B[i + 1024];
	}
	__syncthreads();
	for (int stride = blockDim.x; stride > 0; stride >>= 1) {
		if (t < stride) {
			Shared_mem[t] += Shared_mem[t + stride];

		}
		__syncthreads();
	}
	if (t == 0) {
		d_A[blockIdx.x] = Shared_mem[0];
	}
}



double  gpu_prodct(double* h_a, double* h_b, int d_in_len) {
	double total_sum = 0;
	int block_sz = BLOCKS;
	cudaEvent_t start, stop;
	float elapsedTime;
	double* d_a;
	checkCudaErrors(cudaMalloc(&d_a, sizeof(double) * d_in_len));
	checkCudaErrors(cudaMemcpy(d_a, h_a, sizeof(double) * d_in_len, cudaMemcpyHostToDevice));
	double* d_b;
	checkCudaErrors(cudaMalloc(&d_b, sizeof(double) * d_in_len));
	checkCudaErrors(cudaMemcpy(d_b, h_b, sizeof(double) * d_in_len, cudaMemcpyHostToDevice));

	cudaEventCreate(&start);
	cudaEventRecord(start, 0);
	Dot_Product << <1, block_sz >> > (d_a, d_b, d_in_len);
	cudaDeviceSynchronize();
	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&duration_kernel, start, stop);
	printf("Elapsed time by the Kernel: %f s\n", duration_kernel / 1000);
	duration_kernel /= 1000;


	checkCudaErrors(cudaMemcpy(&total_sum, d_a, sizeof(double), cudaMemcpyDeviceToHost));

	return total_sum;
}

double cpu_product(double *h_a, double* h_b, int h_in_len) {
	double sum = 0;
	for (int i = 0; i < h_in_len; i++)
	{
		sum += (h_a[i] * h_b[i]);
	}
	return sum;
}

int main()
{
	int arraySize;
	clock_t  start;
	do {
		printf("Enter the size of your two vectors between 1 and 2048 inclusive: ");
		scanf("%d", &arraySize);
	} while (arraySize > 2048 || arraySize < 1);

	start = clock();
	double* a = (double*)malloc(arraySize * sizeof(double));

	// if memory cannot be allocated
	if (a == NULL)
	{
		printf("Error! memory not allocated.");
		exit(0);
	}

	double* b = (double*)malloc(arraySize * sizeof(double));

	// if memory cannot be allocated
	if (b == NULL)
	{
		printf("Error! memory not allocated.");
		exit(0);
	}
	clock_t stop = std::clock();
	duration_cpumem = (stop - start) / (double)CLOCKS_PER_SEC;


	srand(time(0));
	for (int i = 0; i < arraySize; i++)
	{
		a[i] = double(rand()) / (double(RAND_MAX / LLONG_MAX) + 1.0);
		b[i] = double(rand()) / (double(RAND_MAX / LLONG_MAX) + 1.0);
	}

	//start CPU implementation
	start = clock();
	double cpu_res = cpu_product(a,b, arraySize);
	stop = std::clock();
	duration_cpu = (stop - start) / (double)CLOCKS_PER_SEC;
	std::cout << "time spent by the CPU without memory: " << duration_cpu << " s" << std::endl;
	duration_cpumem += duration_cpu;
	cout << "time spent by the CPU with memory : " << duration_cpumem << endl;

	//start GPU implementation
	start = std::clock();

	double gpu_res = gpu_prodct(a,b, arraySize);
	cudaDeviceSynchronize();
	stop = std::clock();
	duration_gpu = (stop - start) / (double)CLOCKS_PER_SEC;
	std::cout << "time spent by the GPU: " << duration_gpu << " s" << std::endl;
	printf("CPU result: %lf\n", cpu_res);
	printf("GPU result: %f\n", gpu_res);

	//check the results
	if (gpu_res == cpu_res)
		printf("The two results are the same\n");
	else
		printf("The two results are different\n");

	printf("speedup without memory: %lf\n", duration_cpu / duration_kernel);
	printf("speedup with memory: %lf\n", duration_cpumem / duration_gpu);
	checkCudaErrors(cudaDeviceReset());

	free(a);
	return 0;
}
