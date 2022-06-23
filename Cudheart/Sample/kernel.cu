#include "kernel.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"

#include <stdio.h>
#include <iostream>
#include <chrono>
#include <ctime>

using namespace std;

std::chrono::duration<double> addWithCuda(int* c, const int* a, const int* b, unsigned int size);

__global__ void addKernel(int* c, const int* a, const int* b, int n)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	int v;
	for (int j = 0; j < 500; j++) {
		v = logf(a[i]) + logf(b[i]);
		v = sqrtf(v);
		int s = 50;
		for (int i = 3; i < s; i += i)
		{
			v += 1;
		}
	}
	if (i < n) {
		c[i] = a[i] + b[i];
	}
}

int func()
{
	unsigned long long arraySize = 50000;
	int* a = new int[arraySize];
	int* b = new int[arraySize];
	int* ca = new int[arraySize];
	int* cb = new int[arraySize];

	for (int i = 0; i < arraySize; i++) {
		a[i] = (i + 1);
		b[i] = (i + 1) * 10;
	}
	printf("cuda start\n");

	// Add vectors in parallel.
	std::chrono::duration<double> cuda = addWithCuda(ca, a, b, arraySize);

	//printf("{1,2,3,4,5} + {10,20,30,40,50} = {%f,%f,%f,%f,%f}\n",
		//c[0], c[1], c[2], c[3], c[4]);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaError_t cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	printf("cpp start\n");
	auto start = std::chrono::system_clock::now();
	for (int i = 0; i < arraySize; i++) {
		int v;
		for (int j = 0; j < 500; j++) {
			v = logf(a[i]) + logf(b[i]);
			v = sqrtf(v);
			int s = 50;
			for (int i = 3; i < s; i += i)
			{
				v += 1;
			}
		}
		cb[i] = a[i] + b[i];

		cout << "\ncomputing i " << i;
	}

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	std::time_t end_time = std::chrono::system_clock::to_time_t(end);

	std::cout << "elapsed time: " << elapsed_seconds.count() << "s" << std::endl;

	for (int i = 0; i < arraySize; i++) {
		printf("comparing i %d\n", i);
		if (ca[i] != cb[i]) {
			printf("%d: %d + %d = ", i, a[i], b[i]);
			printf("%d == %d\n", ca[i], cb[i]);
			printf("oof\n");
			break;
		}
	}

	std::cout << "diff is " << elapsed_seconds.count() / cuda.count() << "x" << std::endl;

	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
std::chrono::duration<double> addWithCuda(int* c, const int* a, const int* b, unsigned int size)
{
	int* dev_a = 0;
	int* dev_b = 0;
	int* dev_c = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		cudaFree(dev_c);
		cudaFree(dev_a);
		cudaFree(dev_b);
		std::exit(69);
	}

	cudaDeviceProp stDeviceProp;
	cudaGetDeviceProperties(&stDeviceProp, 0);
	if (!stDeviceProp.unifiedAddressing) {
		fprintf(stderr, "fail");
		cudaFree(dev_c);
		cudaFree(dev_a);
		cudaFree(dev_b);
		std::exit(69);
	}

	printf("Using device %d: %s\n", 0, stDeviceProp.name);

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		cudaFree(dev_c);
		cudaFree(dev_a);
		cudaFree(dev_b);
		std::exit(69);
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		cudaFree(dev_c);
		cudaFree(dev_a);
		cudaFree(dev_b);
		std::exit(69);
	}

	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		cudaFree(dev_c);
		cudaFree(dev_a);
		cudaFree(dev_b);
		std::exit(69);
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		cudaFree(dev_c);
		cudaFree(dev_a);
		cudaFree(dev_b);
		std::exit(69);
	}

	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		cudaFree(dev_c);
		cudaFree(dev_a);
		cudaFree(dev_b);
		std::exit(69);
	}

	int threads = 1024;

	auto start = std::chrono::system_clock::now();

	int N = floor(size / threads) + 1;

	// Launch a kernel on the GPU with one thread for each element.
	addKernel << <N, threads >> > (dev_c, dev_a, dev_b, size);

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	std::time_t end_time = std::chrono::system_clock::to_time_t(end);

	std::cout << "elapsed time: " << elapsed_seconds.count() << "s" << std::endl;

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		cudaFree(dev_c);
		cudaFree(dev_a);
		cudaFree(dev_b);
		std::exit(69);
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %f after launching addKernel!\n", cudaStatus);
		cudaFree(dev_c);
		cudaFree(dev_a);
		cudaFree(dev_b);
		std::exit(69);
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		cudaFree(dev_c);
		cudaFree(dev_a);
		cudaFree(dev_b);
		std::exit(69);
	}

	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return elapsed_seconds;
}