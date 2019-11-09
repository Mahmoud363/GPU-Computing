
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <stdio.h>
#include<cuda.h>
#include<iostream>
#include<cmath>
#include<time.h>
#include<iomanip>
using namespace std;
float duration_kernel;

const int BLOCK_SIZE = 16;
__host__
void matrix_mul_seq(float* a, float* b, float* p, int r1, int w, int c2)
{

	for (int i = 0; i < r1; i++)
		for (int j = 0; j < c2; j++) {
			float sum = 0;
			for (int k = 0; k < w; k++) {
				float x = a[i * w + k];
				float y = b[k * c2 + j];
				sum += x * y;
			}
			p[i * c2 + j] = sum;
		}


}
__global__
void MatrixMulKernel(float* M, float* N,
	float* P, int Width, int r1, int c2)
{
	// Calculate the row index of the P element and M
	int Row = blockIdx.y * blockDim.y + threadIdx.y;
	// Calculate the column index of P and N
	int Col = blockIdx.x * blockDim.x + threadIdx.x;

	float Pvalue = 0;
	// Each thread computes one element of the block sub-matrix
	if (Row < r1 && Col < c2)
	{
		for (int k = 0; k < Width; k++) {
			Pvalue += M[Row * Width + k] * N[k * c2 + Col];

		}



		P[Row * c2 + Col] = Pvalue;
	}
}

void matrix_mul_parallel(float* h_a, float* h_b, float* h_p, int r1, int w, int c2)
{
	int size_a = r1 * w * sizeof(float);
	int size_b = w * c2 * sizeof(float);
	int size_p = r1 * c2 * sizeof(float);
	float* d_a, *d_b, *d_p;

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







	cudaEvent_t star, end;


	cudaEventCreate(&star);
	cudaEventRecord(star, 0);

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
	dim3 dimGrid(ceil(r1 / (float)BLOCK_SIZE), ceil(c2 / (float)BLOCK_SIZE), 1);
	clock_t start = clock();
	MatrixMulKernel << <dimGrid, dimBlock >> > (d_a, d_b, d_p, w, r1, c2);

	cudaEventCreate(&end);
	cudaEventRecord(end, 0);
	cudaEventSynchronize(end);

	cudaEventElapsedTime(&duration_kernel, star, end);
	cout << "Time spent by the Kernel: " << duration_kernel/1000 << " s " << endl;


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
	srand(time(NULL));
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



	float* a = new float[size_a];



	float* b = new float[size_b];

	float* p = new float[size_p];
	float* d_p = new float[size_p];



	// initializing elements of first matrix.
	for (int i = 0; i < r1; i++)
		for (int j = 0; j < w; j++)
		{

			//a[i * w + j] = double(rand()) / (double(RAND_MAX / LLONG_MAX) + 1.0);
			a[i * w + j] =i*j;
		}
	// initializing elements of second matrix.


	for (int i = 0; i < w; i++)
		for (int j = 0; j < c2; j++)
		{

			//b[i * c2 + j] = double(rand()) / (double(RAND_MAX / LLONG_MAX) + 1.0);
			b[i * c2 + j] = i*j;
		}
	// Initializing elements of matrix p to 0.
	for (int i = 0; i < r1; i++)
		for (int j = 0; j < c2; j++)
		{

			p[i * c2 + j] = 0;
		}

	//calling the sequential function
	clock_t start = clock();
	matrix_mul_seq(a, b, p, r1, w, c2);
	clock_t stop = clock();
	double duration_cpu = (double)(stop - start) / CLOCKS_PER_SEC;
	cout << " time spent by cpu in seconds : " << duration_cpu << endl;



	// calling the parallel function 
	start = clock();
	matrix_mul_parallel(a, b, d_p, r1, w, c2);
	stop = clock();
	double duration_device = (double)(stop - start) / CLOCKS_PER_SEC;
	cout << "For Block size: " << BLOCK_SIZE << endl;
	cout << " time spent by the device in seconds : " << duration_device << endl;
	cout << " the speedup/slowdown (timing the kernel only ) is " << duration_cpu / (duration_kernel / 1000) << endl;
	cout << " the speedup/slowdown (timing the device and the memory allocation overheads ) is " << duration_cpu / duration_device << endl;

	int operations_count = ((r1 * w * c2) + (r1 * c2 * (w - 1)));
	cout << " The Performance in GFLOPS = " << operations_count / duration_device << endl;

	return 0;
}