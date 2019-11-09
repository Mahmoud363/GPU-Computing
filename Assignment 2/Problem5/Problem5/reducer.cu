
#include <cmath>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper.h"
using namespace std;

#define BLOCKS 1024



__global__ void reducer( int* g_odata,  int* g_idata,  int len) {
	extern __shared__ unsigned int Shared_mem[];

	// each thread loads one element from global to shared mem
	unsigned int t = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

	//Shared_mem[t] = 0;

	if (i < len)
	{
		Shared_mem[t] = g_idata[i];
	}

	__syncthreads();

	for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
		if (t % (2 * stride) == 0) {
			Shared_mem[t] += Shared_mem[t + stride];
		}
		__syncthreads();
	}

	// by now our shared memory of index 0 should be equal to the sum of the values in this blokc
	// copy it to the blockIdx.x to construct the new array
	// this method is based on the recursive implementation explained in Nvidia documentation by Mark Harris
	if (t == 0)
		g_odata[blockIdx.x] = Shared_mem[0];
}

long long  gpu_reduce( int* d_in,  int d_in_len)
{
	 int total_sum = 0;

	 int block_sz = BLOCKS; 
	

	 int grid_sz = 0;
	if (d_in_len <= block_sz)
	{
		grid_sz = (unsigned int)ceil(float(d_in_len) / float(block_sz));
	}
	else
	{
		grid_sz = d_in_len / block_sz;
		if (d_in_len % block_sz != 0)
			grid_sz++;
	}

	
	 int* d_sum;
	checkCudaErrors(cudaMalloc(&d_sum, sizeof(int) * grid_sz));
	checkCudaErrors(cudaMemset(d_sum, 0, sizeof(int) * grid_sz));


	reducer << <grid_sz, block_sz, sizeof(int) * block_sz >> > (d_sum, d_in, d_in_len);

	if (grid_sz <= block_sz)
	{
		 int* d_total_sum;
		checkCudaErrors(cudaMalloc(&d_total_sum, sizeof(int)));
		checkCudaErrors(cudaMemset(d_total_sum, 0, sizeof(int)));
		reducer << <1, block_sz, sizeof(int) * block_sz >> > (d_total_sum, d_sum, grid_sz);
		//reduce4<<<1, block_sz, sizeof(int) * block_sz>>>(d_total_sum, d_sum, grid_sz);
		checkCudaErrors(cudaMemcpy(&total_sum, d_total_sum, sizeof( int), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaFree(d_total_sum));
	}
	else
	{
		 int* d_in_sums;
		checkCudaErrors(cudaMalloc(&d_in_sums, sizeof(int) * grid_sz));
		checkCudaErrors(cudaMemcpy(d_in_sums, d_sum, sizeof(int) * grid_sz, cudaMemcpyDeviceToDevice));
		total_sum = gpu_reduce(d_in_sums, grid_sz);
		checkCudaErrors(cudaFree(d_in_sums));
	}

	checkCudaErrors(cudaFree(d_sum));
	return total_sum;
}
