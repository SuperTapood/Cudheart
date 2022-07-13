#include "kernel.cuh"
#include "UtilKernel.cuh"

using namespace std;

int func()
{
	long long arraySize = 500000;
	int nBytes = sizeof(int) * arraySize;
	int* a, * b, * ca, * hostRef, * ra, * rb, * rc, * cc;
	a = (int*)malloc(nBytes);
	b = (int*)malloc(nBytes);
	ra = (int*)malloc(nBytes);
	rb = (int*)malloc(nBytes);
	rc = (int*)malloc(nBytes);
	cc = (int*)malloc(nBytes);
	hostRef = (int*)malloc(nBytes);

	cudaError_t err = cudaMallocManaged(&a, nBytes, cudaMemAttachGlobal);
	if (err != cudaSuccess)
	{
		printf("Not support cudaMallocManaged!\n");
		goto EXIT;
	}

	err = cudaMallocManaged(&b, nBytes, cudaMemAttachGlobal);
	if (err != cudaSuccess)
	{
		printf("Not support cudaMallocManaged!\n");
		goto EXIT;
	}

	err = cudaMallocManaged(&ca, nBytes, cudaMemAttachGlobal);
	if (err != cudaSuccess)
	{
		printf("Not support cudaMallocManaged!\n");
		goto EXIT;
	}

	for (int i = 0; i < arraySize; i++) {
		a[i] = (i + 1);
		b[i] = (i + 1) * 10;
		ra[i] = (i + 1);
		rb[i] = (i + 1) * 10;
		hostRef[i] = 0;
	}

	//printf("cuda start\n");

	// Add vectors in parallel.
	std::chrono::duration<double> cuda = addWithCudaUni(ca, a, b, arraySize);

	//printf("{1,2,3,4,5} + {10,20,30,40,50} = {%f,%f,%f,%f,%f}\n",
		//c[0], c[1], c[2], c[3], c[4]);

	cudaDeviceSynchronize();

	//printf("cuda memcopy start\n");
	std::chrono::duration<double> cudaReg = addWithCuda(cc, ra, rb, arraySize);

	//printf("cpp start\n");
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

	//std::cout << "elapsed time: " << elapsed_seconds_uni.count() << "s" << std::endl;

	//printf("start raw cpp\n");
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

	//std::cout << "elapsed time: " << elapsed_seconds_raw.count() << "s" << std::endl;

	for (int i = 0; i < arraySize; i++) {
		if (ca[i] != hostRef[i] || ca[i] != rc[i] || ca[i] != cc[i]) {
			printf("%d: %d + %d = (%d + %d) = ", i, a[i], b[i], ra[i], rb[i]);
			printf("%d == %d == %d\n", ca[i], hostRef[i], rc[i]);
			printf("oof\n");
			break;
		}
	}

	double diff = elapsed_seconds_uni.count() / cuda.count();

	std::cout << "zero copy cuda is:\n" << elapsed_seconds_uni.count() / cuda.count() << " times faster than unified c++\n" << elapsed_seconds_raw.count() / cuda.count() << " times faster than raw c++\n" << cuda.count() / cudaReg.count() << " times slower than regular cuda with memory copy (" << cudaReg.count() << " vs " << cuda.count() << ")\n";

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaError_t cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;

EXIT:
	cudaDeviceReset();

	system("pause");
	return 0;
}