
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <stdio.h>
#include<cuda.h>
#include<iostream>
#include<cmath>
#include<time.h>
#define TILE_WIDTH 4//block size


double duration_gpu, duration_cpu, duration_kernel, duration_cpumem;
using namespace std;
__host__
void matrix_mul_seq(double* a, double* b, double* p, int r1, int w, int c2)
{
	clock_t start = clock();
	for (int i = 0; i < r1; i++)
		for (int j = 0; j < c2; j++) {
			double sum = 0;
			for (int k = 0; k < w; k++) {
				double x = a[i *w + k];
				double y = b[k * c2 + j];
				sum += x * y;
			}
			p[i * c2 + j] = sum;
		}

	clock_t stop = clock();
	duration_cpu = (double)(stop - start) / CLOCKS_PER_SEC;
	cout << " time spent by cpu in seconds : " << duration_cpu << endl;

}
__global__
void MatrixMulKernel(double* M, double* N,
	double* P, int Width, int r1, int c2)
{
	// Calculate the row index of the P element and M
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	// Calculate the column index of P and N
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ double Ms[TILE_WIDTH][TILE_WIDTH];
	__shared__ double Ns[TILE_WIDTH][TILE_WIDTH];
	for (int Row = i; Row < gridDim.y * blockDim.y * 2 + threadIdx.y; Row += gridDim.y * blockDim.y) {
		for (int Col = j; Col < gridDim.x * blockDim.x * 2 + threadIdx.x; Col += gridDim.x * blockDim.x) {
			double Pvalue = 0;
			// Each thread computes one element of the block sub-matrix
			for (int k = 0; k < (Width + TILE_WIDTH - 1) / TILE_WIDTH; ++k) {

				if (k * TILE_WIDTH + threadIdx.x < Width && Row < r1)
					Ms[threadIdx.y][threadIdx.x] = M[Row * Width + k * TILE_WIDTH + threadIdx.x];
				else
					Ms[threadIdx.y][threadIdx.x] = 0.0;


				if (k * TILE_WIDTH + threadIdx.y < Width && Col < c2)
					Ns[threadIdx.y][threadIdx.x] = N[(k * TILE_WIDTH + threadIdx.y) * c2 + Col];
				else
					Ns[threadIdx.y][threadIdx.x] = 0.0;

				__syncthreads();

				for (int n = 0; n < TILE_WIDTH; n++) {
					//if (n + (k * TILE_WIDTH) < Width)
					Pvalue += Ms[threadIdx.y][n] * Ns[n][threadIdx.x];
				}

				__syncthreads();
			}

			if (Row < r1 && Col < c2)
			{

				P[Row * c2 + Col] = Pvalue;
			}
		}
	}
}

void matrix_mul_parallel(double* h_a, double* h_b, double* h_p, int r1, int w, int c2)
{
	int size_a = r1 * w * sizeof(double);
	int size_b = w * c2 * sizeof(double);
	int size_p = r1 * c2 * sizeof(double);
	double* d_a, *d_b, *d_p;

	cudaError_t err = cudaMalloc((void**)&d_a, size_a);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	err = cudaMalloc((void**)&d_b, size_b);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}


	err = cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}


	err = cudaMalloc((void**)&d_p, size_p);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}




	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
	dim3 dimGrid(ceil(c2 / float(dimBlock.x * 2)), (ceil(r1 / float(dimBlock.y * 2))));
	clock_t start = clock();
	MatrixMulKernel << <dimGrid, dimBlock >> > (d_a, d_b, d_p, w, r1, c2);
	cudaDeviceSynchronize();
	clock_t stop = clock();
	duration_kernel = (double)(stop - start) / CLOCKS_PER_SEC;
	cout << " time spent by the kernel in seconds : " << duration_kernel << endl;
	err = cudaMemcpy(h_p, d_p, size_p, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	cudaFree(d_a); cudaFree(d_b); cudaFree(d_p);



}
int main()
{
	srand(unsigned(time(0)));
	int r1, w, c2;
	cout << "Enter rows for first matrix: ";
	cin >> r1;
	cout << "Enter columns of first matrix which is the same as rows for second matrix: ";
	cin >> w;
	cout << "Enter columns for second matrix: ";
	cin >> c2;

	int size_a = r1 * w;
	int size_b = w * c2;
	int size_p = r1 * c2;


	clock_t start = clock();
	double* a = new double[size_a];


	double* b = new double[size_b];

	double* p = new double[size_p];
	double* d_p = new double[size_p];
	clock_t stop = clock();
	duration_cpumem = (double)(stop - start) / CLOCKS_PER_SEC;




	// initializing elements of first matrix.
	for (int i = 0; i < r1; i++)
		for (int j = 0; j < w; j++)
		{

			a[i * w + j] = double(rand()) / (double(RAND_MAX / LLONG_MAX) + 1.0);
		}
	// initializing elements of second matrix.
	srand(unsigned(time(0)));

	for (int i = 0; i < w; i++)
		for (int j = 0; j < c2; j++)
		{

			b[i * c2 + j] = double(rand()) / (double(RAND_MAX / LLONG_MAX) + 1.0);
		}
	// Initializing elements of matrix p to 0.
	for (int i = 0; i < r1; i++)
		for (int j = 0; j < c2; j++)
		{

			p[i * c2 + j] = 0;
		}

	//calling the sequential function
	matrix_mul_seq(a, b, p, r1, w, c2);
	duration_cpumem += duration_cpu;
	cout << " time spent by the CPU with memory : " << duration_cpumem << endl;

	// calling the parallel function 
	start = clock();
	matrix_mul_parallel(a, b, d_p, r1, w, c2);
	cudaDeviceSynchronize();
	stop = clock();
	duration_gpu = (double)(stop - start) / CLOCKS_PER_SEC;
	cout << " time spent by the kernel in seconds : " << duration_gpu << endl;
	unsigned long long  int counter = 0;;
	for (int i = 0; i < r1; ++i)
		for (int j = 0; j < c2; ++j)
		{
			counter += (d_p[i * c2 + j] != p[i * c2 + j]);

		}
	printf("There are %ld different elements\n", counter);
	printf("speedup without memory: %lf\n", duration_cpu / duration_kernel);
	printf("speedup with memory: %lf\n", duration_cpumem / duration_gpu);
	system("pause");
	return 0;
}