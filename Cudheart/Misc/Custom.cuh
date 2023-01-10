#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"
#include <stdio.h>
#include <iostream>
#include <chrono>
#include <ctime>
#include <device_functions.h>
#include <iostream>

using namespace std;

template <typename A, typename B, typename C>
__global__ void add(A* c, B* a, C* b) {
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
	printf("Hello thread %d with values: %f, %d, %f\n", i, a[i], b[i], c[i]);
}

void testCustom() {
	float a[] = { 1.5, 2.5, 3.5, 4.5 };
	int b[] = { 7, 12, 54, 578 };
	double* c = (double*)malloc(sizeof(double) * 4);
	float* dev_a;
	int* dev_b;
	double* dev_c;
	cudaError_t cudaStatus;

	cudaMalloc(&dev_a, sizeof(float) * 4);
	cudaMalloc(&dev_b, sizeof(int) * 4);
	cudaMalloc(&dev_c, sizeof(double) * 4);

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, 4 * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		cudaFree(dev_c);
		cudaFree(dev_a);
		cudaFree(dev_b);
		std::exit(69);
	}

	cudaStatus = cudaMemcpy(dev_b, b, 4 * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		cudaFree(dev_c);
		cudaFree(dev_a);
		cudaFree(dev_b);
		std::exit(69);
	}

	add << <1, 4 >> > (dev_c, dev_a, dev_b);
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaDeviceSynchronize();

	cudaStatus = cudaMemcpy(c, dev_c, 4 * sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		cudaFree(dev_c);
		cudaFree(dev_a);
		cudaFree(dev_b);
		std::exit(69);
	}

	cout << "True is " << true << endl;
	for (int i = 0; i < 4; i++) {
		cout << (c[i] == a[i] + b[i]) << endl;
	}

	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);
}