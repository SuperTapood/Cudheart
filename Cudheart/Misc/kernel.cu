#include "kernel.cuh"
#include "UtilKernel.cuh"

using namespace std;

/// <summary>
/// compare regular cuda, unified cuda, unified c++, and regular c++ and present the results.
/// </summary>
void func()
{
	long long arraySize = 50000;
	int nBytes = sizeof(int) * arraySize;
	int* a, * b, * ca, * hostRef, * ra, * rb, * rc, * cc;
	// unified input and output arrays
	a = (int*)malloc(nBytes);
	b = (int*)malloc(nBytes);
	hostRef = (int*)malloc(nBytes);
	// raw c arrays
	ra = (int*)malloc(nBytes);
	rb = (int*)malloc(nBytes);
	rc = (int*)malloc(nBytes);
	// output raw cuda array
	cc = (int*)malloc(nBytes);

	// try to allocate the memory on the cuda unified whatever
	cudaError_t err = cudaMallocManaged(&a, nBytes, cudaMemAttachGlobal);
	if (err != cudaSuccess)
	{
		printf("Not support cudaMallocManaged!\n");
		cudaDeviceReset();
		system("pause");
		return;
	}

	err = cudaMallocManaged(&b, nBytes, cudaMemAttachGlobal);
	if (err != cudaSuccess)
	{
		printf("Not support cudaMallocManaged!\n");
		cudaDeviceReset();
		system("pause");
		return;
	}

	err = cudaMallocManaged(&ca, nBytes, cudaMemAttachGlobal);
	if (err != cudaSuccess)
	{
		printf("Not support cudaMallocManaged!\n");
		cudaDeviceReset();
		system("pause");
		return;
	}

	for (int i = 0; i < arraySize; i++) {
		a[i] = (i + 1);
		b[i] = (i + 1) * 10;
		ra[i] = (i + 1);
		rb[i] = (i + 1) * 10;
		hostRef[i] = 0;
	}

	// Add vectors in parallel.
	std::chrono::duration<double> cuda = addWithCudaUni(ca, a, b, arraySize);

	cudaDeviceSynchronize();

	std::chrono::duration<double> cudaReg = addWithCuda(cc, ra, rb, arraySize);

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
		hostRef[i] = a[i] + b[i];
	}

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds_uni = end - start;
	std::time_t end_time = std::chrono::system_clock::to_time_t(end);

	start = std::chrono::system_clock::now();
	for (int i = 0; i < arraySize; i++) {
		int v;
		for (int j = 0; j < 500; j++) {
			v = logf(ra[i]) + logf(rb[i]);
			v = sqrtf(v);
			int s = 50;
			for (int i = 3; i < s; i += i)
			{
				v += 1;
			}
		}
		rc[i] = ra[i] + rb[i];
	}

	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds_raw = end - start;
	end_time = std::chrono::system_clock::to_time_t(end);

	for (int i = 0; i < arraySize; i++) {
		if (ca[i] != hostRef[i] || ca[i] != rc[i] || ca[i] != cc[i]) {
			printf("%d: %d + %d = (%d + %d) = ", i, a[i], b[i], ra[i], rb[i]);
			printf("%d == %d == %d\n", ca[i], hostRef[i], rc[i]);
			printf("oof\n");
			break;
		}
	}

	double diff = elapsed_seconds_uni.count() / cuda.count();

	printf("on a %d elements array with semi complex math, zero copy (unified) cuda is:\n", arraySize);
	printf("%.4f%% the speed of regular c++ (%.4fs vs %.4fs)\n", ((elapsed_seconds_raw.count() / cuda.count()) * 100), elapsed_seconds_raw.count(), cuda.count());
	printf("%.4f%% the speed of unified c++ (%.4fs vs %.4fs)\n", ((elapsed_seconds_uni.count() / cuda.count()) * 100), elapsed_seconds_uni.count(), cuda.count());
	printf("%.4f%% the speed of regular cuda (%.4fs vs %.4fs)\n", ((cudaReg.count() / cuda.count()) * 100), cudaReg.count(), cuda.count());
	printf("(regular c++ is %.4f%% faster than unified c++) (%.4fs vs %.4fs)\n", ((elapsed_seconds_uni.count() / elapsed_seconds_raw.count()) * 100), elapsed_seconds_uni.count(), elapsed_seconds_raw.count());

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaError_t cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return;
	}

	return;
}