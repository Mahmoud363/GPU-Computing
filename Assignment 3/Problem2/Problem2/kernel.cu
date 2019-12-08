
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<string>
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

float duration_gpu, duration_cpu, duration_kernel, duration_cpumem;

__global__ void scan(unsigned char* X, unsigned char* Y, unsigned char* Z, int colsize, int rowsize)
{
	__shared__ double T[256];
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int index = j * colsize + i;
	if (index < (colsize * rowsize))
	{
		if (i < colsize && j < rowsize) {
			T[threadIdx.x] = Y[index]; // error 
		}
		else T[threadIdx.x] = 0;
		//the code below performs iterative scan on T
		for (int stride = 1; stride < blockDim.x; stride *= 2) {
			float temp = 0;
			if (threadIdx.x >= stride) {
				__syncthreads();
				if (threadIdx.x - stride >= 0)
					temp = T[threadIdx.x] + T[threadIdx.x - stride];
				__syncthreads();
				T[threadIdx.x] = temp;
			}
		}
		X[index] = T[threadIdx.x];
		//printf("%d\n", X[index]);

		if (threadIdx.x == blockDim.x - 1) {
			//printf("%d\n", T[threadIdx.x]);
			Z[blockIdx.y * blockDim.x + blockIdx.x] = T[threadIdx.x];
		}
	}
}

__global__ void fix(unsigned char* X, unsigned char* Y, int colsize, int rowsize)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int index = j * colsize + i;
	int aux = blockIdx.y * blockDim.x + blockIdx.x - 1;
	if (index < colsize*rowsize) {
		if (blockIdx.x) {
			int temp = Y[aux];
			X[index] += temp;
		}
	}
}

__global__ void transpose(unsigned char *input, unsigned char *output, int width, int height)
{

	__shared__ unsigned char temp[16][16];

	int xIndex = blockIdx.x*blockDim.x + threadIdx.x;
	int yIndex = blockIdx.y*blockDim.y + threadIdx.y;

	if ((xIndex < width) && (yIndex < height)) {
		int id_in = yIndex * width + xIndex;
		temp[threadIdx.y][threadIdx.x] = input[id_in];
	}
	__syncthreads();

	xIndex = blockIdx.y * blockDim.y + threadIdx.x;
	yIndex = blockIdx.x * blockDim.x + threadIdx.y;

	if ((xIndex < height) && (yIndex < width)) {
		int id_out = yIndex * height + xIndex;
		output[id_out] = temp[threadIdx.x][threadIdx.y];
	}
}

void gpu_kernel(unsigned char* h_in, int height, int width){

	unsigned char* d_in, *d_out, *aux, *auxout, *aux2, *auxout2,*d_in2,*d_fin;
	int n = height * width;
	checkCudaErrors(cudaMalloc(&d_in, sizeof(unsigned char) *n));
	checkCudaErrors(cudaMalloc(&d_fin, sizeof(unsigned char) *n));
	checkCudaErrors(cudaMalloc(&aux, sizeof(unsigned char) * 512 * height));
	checkCudaErrors(cudaMalloc(&auxout, sizeof(unsigned char) *512*height));
	checkCudaErrors(cudaMalloc(&d_out, sizeof(unsigned char) * n));
	checkCudaErrors(cudaMemcpy(d_in, h_in, sizeof(unsigned char) * n, cudaMemcpyHostToDevice));


	cudaEvent_t start, stop;
	float elapsedTime;

	cudaEventCreate(&start);
	cudaEventRecord(start, 0);
	dim3 grids(512, height);
	dim3 grids3(512, width);
	dim3 threads(256);
	dim3 grids2((ceil(width / 16.0), ceil(height / 166.0), 1));
	dim3 threads2(16,16);
	scan << <grids, threads >> > (d_out, d_in,aux, width, height);
	scan << <1, 512 >> > (auxout, aux,NULL, width, 1);
	fix << <grids, threads>> > (d_out, auxout,width, height);

	transpose << <grids2, threads2 >> > (d_out, d_out, width, height);
	
	scan << <grids3, threads >> > (d_fin, d_out, aux, height, width);
	scan << <1, 512 >> > (auxout, aux, NULL, height, 1);
	fix << <grids3, threads >> > (d_fin, auxout, height, width);
	transpose << < grids2, threads2 >> > (d_fin, d_fin, height, width);
	
	unsigned char *h_img;
	h_img = (unsigned char *)malloc(height * width * sizeof(unsigned char));
	checkCudaErrors(cudaMemcpy(h_img, d_fin, sizeof(unsigned char) * n, cudaMemcpyDeviceToHost));

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			cout << (int)h_img[i * width + j] << " ";
		}
		cout << endl;
	}

	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&duration_kernel, start, stop);
	printf("Elapsed time by the Kernel: %f s\n", duration_kernel / 1000);
	duration_kernel /= 1000;

	
	//unsigned long long count = 0;
	//for (int i = 0; i < height; i++) {
	//	for (int j = 0; j < width; j++) {
	//		if (h_img[i * width + j] != h_in[i * width + j])
	//			count++;
	//	}
	//}
	//cout << count << "  " << n << endl;
	/*CImg<unsigned char> image_out(width, height, 1, 1, 0);
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			image_out(j, i, 0, 0) = h_img[i * width + j];
		}
	}
	image_out.save("out.jpg");*/



}

double cpu_reduce(double *h_in, int h_in_len) {
	double sum = 0;
	for (int i = 0; i < h_in_len; i++)
	{
		sum += h_in[i];
	}
	return sum;
}
void prefixSum2D(unsigned char *a,int r, int c)
{
	unsigned char *psa;
	psa = (unsigned char *)malloc(r *c* sizeof(unsigned char));
	psa[0] = a[0];

	// Filling first row and first column 
	for (int i = 1; i < c; i++)
		psa[i] = psa[i - 1] + a[i];
	for (int i = 0; i < r; i++)
		psa[i*c] = psa[(i - 1)*c] + a[i*c];

	// updating the values in the cells 
	// as per the general formula 
	for (int i = 1; i < r; i++) {
		for (int j = 1; j < c; j++)

			// values in the cells of new 
			// array are updated 
			psa[i*c + j] = psa[(i - 1)*c + j] + psa[i*c + (j - 1)];
			
	}

	// displaying the values of the new array 
	for (int i = 0; i < r; i++) {
		for (int j = 0; j < c; j++)
			cout << (int)psa[i*c+j] << " ";
		cout << "\n";
	}
}
int main()
{
	/*
	string path;
	cout << "Enter the image path: ";
	cin >> path;
	cout << "Enter the number of the operation: \n";
	printf("1 for blur. 2 for emboss. 3 for outline. 4 for sharpen\n 5 for left sobel. 6 for right sobel. 7 for top sobel. 8 for bottom sobel\n");
	int type;
	cin >> type;

	CImg<unsigned char> image(path.c_str());
	image.channel(0);*/
	
	//int height = image.height();
	//int width = image.width();
	int height = 5;
	int width = 5;
	unsigned char *h_img;
	h_img = (unsigned char *)malloc(height *width * sizeof(unsigned char));

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			//h_img[i * width + j] = (unsigned char)image(j, i, 0, 0);
			h_img[i * width + j] = 10;
		}
	}

	int arraySize;
	clock_t  start;

	gpu_kernel(h_img, height, width);
	prefixSum2D(h_img, height, width);

	/*switch (type)
	{
	case 1:gpu_kernel(h_img, height, width, blur);
		break;
	case 2:gpu_kernel(h_img, height, width, emboss);
		break;
	case 3:gpu_kernel(h_img, height, width, outline);
		break;
	case 4:gpu_kernel(h_img, height, width, sharpen);
		break;
	case 5:gpu_kernel(h_img, height, width, left);
		break;
	case 6:gpu_kernel(h_img, height, width, right);
		break;
	case 7:gpu_kernel(h_img, height, width, top);
		break;
	case 8:gpu_kernel(h_img, height, width, bottom);
		break;
	default:
		printf("Invalid operation");
		break;
	}*/
	free(h_img);



	return 0;
}
