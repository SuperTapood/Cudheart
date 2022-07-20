#include "UtilKernel.cuh"

using namespace std;

std::chrono::duration<double> addWithCudaUni(int* c, const int* a, const int* b, unsigned int size);

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

// Helper function for using CUDA to add vectors in parallel.
std::chrono::duration<double> addWithCudaUni(int* c, const int* a, const int* b, unsigned int size)
{
	int threads = 1024;

	auto start = std::chrono::system_clock::now();

	int N = floor(size / threads) + 1;

	// Launch a kernel on the GPU with one thread for each element.
	addKernel << <N, threads >> > (c, a, b, size);

	// Check for any errors launching the kernel
	auto cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		std::exit(69);
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		std::exit(69);
	}

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	std::time_t end_time = std::chrono::system_clock::to_time_t(end);

	//std::cout << "elapsed time: " << elapsed_seconds.count() << "s" << std::endl;

	return elapsed_seconds;
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

	auto start = std::chrono::system_clock::now();

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

	int N = floor(size / threads) + 1;

	// Launch a kernel on the GPU with one thread for each element.
	addKernel << <N, threads >> > (dev_c, dev_a, dev_b, size);

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
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
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

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	std::time_t end_time = std::chrono::system_clock::to_time_t(end);

	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	//std::cout << "elapsed time: " << elapsed_seconds.count() << "s" << std::endl;

	return elapsed_seconds;
}