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

__global__ void reducer_min_divergence(double* d_in, int len) {
	extern __shared__ double Shared_mem[BLOCKS];
	int t = threadIdx.x;
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	Shared_mem[t] = 0;
	if (i < len)
	{
		Shared_mem[t] = d_in[i];
	}

	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		__syncthreads();
		if (t < stride && t + stride < 1024) {
			Shared_mem[t] += Shared_mem[t + stride];

		}
	}
	__syncthreads();
	if (t == 0) {
		d_in[blockIdx.x] = Shared_mem[0];
	}
}


double  gpu_reduce(double* h_in, int d_in_len)
{
	double total_sum = 0;

	int block_sz = BLOCKS;


	int grid_sz = 0;
	if (d_in_len <= block_sz)
	{
		grid_sz = (unsigned int)ceil(double(d_in_len) / double(block_sz));
	}
	else
	{
		grid_sz = d_in_len / block_sz;
		if (d_in_len % block_sz != 0)
			grid_sz++;
	}

	double* d_in;
	checkCudaErrors(cudaMalloc(&d_in, sizeof(double) * d_in_len));
	checkCudaErrors(cudaMemcpy(d_in, h_in, sizeof(double) * d_in_len, cudaMemcpyHostToDevice));
	int n = d_in_len;

	cudaEvent_t start, stop;
	float elapsedTime;

	cudaEventCreate(&start);
	cudaEventRecord(start, 0);
	while (n > 1)
	{
		reducer_min_divergence << <grid_sz, block_sz >> > (d_in, n);
		//cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());
		n = ceil(n / 1024.0);
	}
	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&duration_kernel, start, stop);
	printf("Elapsed time by the Kernel: %f s\n", duration_kernel / 1000);
	duration_kernel /= 1000;


	checkCudaErrors(cudaMemcpy(&total_sum, d_in, sizeof(double), cudaMemcpyDeviceToHost));

	return total_sum;
}

double cpu_reduce(double *h_in, int h_in_len) {
	double sum = 0;
	for (int i = 0; i < h_in_len; i++)
	{
		sum += h_in[i];
	}
	return sum;
}

int main()
{
	int arraySize;
	clock_t  start;
	printf("Enter the size of your array: ");
	scanf("%d", &arraySize);


	start = clock();
	double* a = (double*)malloc(arraySize * sizeof(double));
	clock_t stop = std::clock();
	duration_cpumem = (stop - start) / (double)CLOCKS_PER_SEC;


	// if memory cannot be allocated
	if (a == NULL)
	{
		printf("Error! memory not allocated.");
		exit(0);
	}


	srand(time(0));
	for (int i = 0; i < arraySize; i++)
	{
		a[i] = double(rand()) / (double(RAND_MAX/LLONG_MAX) + 1.0);
	}

	//start CPU implementation
	start = clock();
	double cpu_res = cpu_reduce(a, arraySize);
	stop = std::clock();
	duration_cpu = (stop - start) / (double)CLOCKS_PER_SEC;
	std::cout << "time spent by the CPU without memory: " << duration_cpu << " s" << std::endl;
	duration_cpumem += duration_cpu;
	cout << " time spent by the CPU with memory : " << duration_cpumem << endl;

	//start GPU implementation
	start = std::clock();

	double gpu_res = gpu_reduce(a, arraySize);
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
