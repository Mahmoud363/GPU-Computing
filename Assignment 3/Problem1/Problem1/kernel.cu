
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "CImg.h"
using namespace cimg_library;
#include<math.h>
#include<time.h>
#include<stdlib.h>
#include <stdio.h>
#include <ctime>
#include <cmath>
#include "helper.h"
using namespace std;

#define TILE_WIDTH 8
#define MAX_MASK_WIDTH 3
__constant__ double M[MAX_MASK_WIDTH][MAX_MASK_WIDTH];


float duration_gpu, duration_cpu, duration_kernel, duration_cpumem;

__global__ void ConvolutionKernel(unsigned char *P, unsigned char *N, int height, int width) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int row_o = blockIdx.y * TILE_WIDTH + ty;
	int col_o = blockIdx.x * TILE_WIDTH + tx;
	if (row_o < height && col_o < width) {
		int row_i = row_o - MAX_MASK_WIDTH / 2;
		int col_i = col_o - MAX_MASK_WIDTH / 2;
		__shared__ float N_ds[TILE_WIDTH][TILE_WIDTH];
		if (row_o * width + col_o < height*width)
			N_ds[ty][tx] = N[row_o * width + col_o];




		// Wait until all tile elements are loaded
		__syncthreads();

		double Pvalue = 0.0f;
		int This_tile_col_start_point = blockIdx.x * blockDim.x;
		int Next_tile_col_start_point = (blockIdx.x + 1) * blockDim.x;
		int This_tile_row_start_point = blockIdx.y * blockDim.y;
		int Next_tile_row_start_point = (blockIdx.y + 1) * blockDim.y;

		if (ty < TILE_WIDTH && tx < TILE_WIDTH) {
			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {
					int row_index = row_i + i; int col_index = col_i + j;
					if ((row_index >= This_tile_row_start_point) && (row_index < Next_tile_row_start_point)&&row_index<height) {
						if ((col_index >= This_tile_col_start_point) && (col_index < Next_tile_col_start_point)&&col_index<width)
							Pvalue += N_ds[threadIdx.y + i - (MAX_MASK_WIDTH / 2)][threadIdx.x + j - (MAX_MASK_WIDTH / 2)] * M[i][j];
						else if (row_index >= height && col_i >= width) // 1
							Pvalue +=
							N[(height - 1) * width + (width - 1)] * M[i][j];
						else if (row_index >= height && col_index < width &&
							col_index >= 0) // 2
							Pvalue +=
							N[(height - 1) * width + col_index] * M[i][j];
						else if (row_index >= height && col_index < 0) // 3
							Pvalue += N[(height - 1) * width + 0] * M[i][j];
						else if (row_index < height && row_index >= 0 &&
							col_index < 0) // 4
							Pvalue += N[row_index * width + 0] * M[i][j];
						else if (row_index < 0 && col_index < 0) // 5
							Pvalue += N[0 * width + 0] * M[i][j];
						else if (row_index < 0 && col_index < width &&
							col_index >= 0) // 6
							Pvalue += N[0 * width + col_index] * M[i][j];
						else if (row_index < 0 && col_index >= width) // 7
							Pvalue += N[0 * width + (width - 1)] * M[i][j];
						else if (row_index >= 0 && row_index < height &&
							col_index >= width) // 8
							Pvalue += N[row_index * width + (width - 1)] * M[i][j];
						else
							Pvalue += N[row_index * width + col_index] * M[i][j];
					}

					else if (row_index >= height && col_index >= width) // 1
						Pvalue += N[(height - 1) * width + (width - 1)] * M[i][j];
					else if (row_index >= height && col_index < width &&
						col_index >= 0) // 2
						Pvalue +=
						N[(height - 1) * width + col_index] * M[i][j];
					else if (row_index >= height && col_index < 0) // 3
						Pvalue +=
						N[(height - 1) * width + 0] * M[i][j];
					else if (row_index < height && row_index >= 0 && col_index < 0) // 4
						Pvalue +=
						N[row_index* width + 0] * M[i][j];
					else if (row_index < 0 && col_index < 0) // 5
						Pvalue +=
						N[0 * width + 0] * M[i][j];
					else if (row_index < 0 && col_index < width && col_index >= 0) // 6
						Pvalue +=
						N[0 * width + col_index] * M[i][j];
					else if (row_index < 0 && col_index >= width) // 7
						Pvalue +=
						N[0 * width + (width - 1)] * M[i][j];
					else if (row_index >= 0 && row_index < height &&
						col_index >= width) // 8
						Pvalue +=
						N[row_index * width + (width - 1)] * M[i][j];
					else
						Pvalue += N[row_index * width + col_index] * M[i][j];

				}
			}
			if (row_o < height && col_o < width) {
				P[row_o * width + col_o] = Pvalue;
			}

		}
	}
}



void gpu_kernel(unsigned char* h_in, int height, int width, double filter[][3])
{
	/*
	 double filter[3][3] = {
		{0.0625,.125,0.0625},
		{0.125,0.25,0.125},
		{0.0625,0.125,.0625}
	};
  */
	
	cudaMemcpyToSymbol(M, filter, 3 * 3 * sizeof(double));
	unsigned char* d_in, *d_out;
	int n = height * width;
	checkCudaErrors(cudaMalloc(&d_in, sizeof(unsigned char) *n));
	checkCudaErrors(cudaMalloc(&d_out, sizeof(unsigned char) * n));
	checkCudaErrors(cudaMemcpy(d_in, h_in, sizeof(unsigned char) * n, cudaMemcpyHostToDevice));


	cudaEvent_t start, stop;
	float elapsedTime;

	cudaEventCreate(&start);
	cudaEventRecord(start, 0);
	dim3 grids(ceil(width / 8.0), ceil(height / 8.0));
	dim3 threads(8, 8);
	ConvolutionKernel << <grids, threads >> > (d_out, d_in, height, width);

	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&duration_kernel, start, stop);
	printf("Elapsed time by the Kernel: %f s\n", duration_kernel / 1000);
	duration_kernel /= 1000;

	unsigned char *h_img;

	h_img = (unsigned char *)malloc(height * width * sizeof(unsigned char));
	checkCudaErrors(cudaMemcpy(h_img, d_out, sizeof(unsigned char) * n, cudaMemcpyDeviceToHost));
	unsigned long long count = 0;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (h_img[i * width + j] != h_in[i * width + j])
				count++;
		}
	}
	cout << count << "  " << n << endl;
	/*for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			cout << (int)h_img[i * width + j] << " ";
		}
		cout << endl;
	}*/
	CImg<unsigned char> image_out(width, height, 1, 1, 0);
	for (int i = 0; i < height; i++) {
	  for (int j = 0; j < width; j++) {
		image_out(j, i, 0, 0) = h_img[i * width + j];
		//cout << h_img[i * width + j] << " ";
	  }
	 // cout << endl;
	}
	image_out.save("out.jpg");



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
	double blur[3][3] = {
	   {0.0625,.125,0.0625},
	   {0.125,0.25,0.125},
	   {0.0625,0.125,.0625}
	};
	double emboss[3][3] = {
	   {-2,-1,0},
	   {-1,1,1},
	   {0,1,2}
	};
	double outline[3][3] = {
	   {-1,-1,-1},
	   {-1,8,-1},
	   {-1,-1,-1}
	};
	double sharpen[3][3] = {
	   {0,-1,0},
	   {-1,5,-1},
	   {0,-1,0}
	};
	double blur[3][3] = {
	   {-2,-1,0},
	   {-1,1,1},
	   {0,1,2}
	};
	CImg<unsigned char> image("fl.jpg");
	image.channel(0);
	unsigned char *h_img;

	h_img = (unsigned char *)malloc(image.height() *image.width() * sizeof(unsigned char));
	int height = image.height();
	int width  = image.width();
	//int height = 10;
	//int width = 10;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			h_img[i * width+ j ] = (unsigned char)image(j, i, 0,0 );
			//h_img[i * width + j] = 10;
		}
	}

	int arraySize;
	clock_t  start;



	gpu_kernel(h_img, height, width);




	// if memory cannot be allocated
	if (h_img == NULL)
	{
		printf("Error! memory not allocated.");
		exit(0);
	}
}
