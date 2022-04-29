#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../Exceptions/CudaException.cuh"

#include <stdio.h>


template <typename T>
class ContainerA {
private:
	T* m_ptrA;
	cudaError_t cudaStatus;
	int m_size;

	bool cool;

public:
	T* devA;

private:
	void release() {
		cudaFree((void*)devA);
	}

	void checkStatus(cudaError_t status, string func) {
		if (status != cudaSuccess) {
			release();
			Cudheart::Exceptions::CudaException(status, func).raise();
		}
	}

public:
	~ContainerA() {
		if (!cool) {
			coolDown();
		}
	}

	void warmUp(T* a, int size) {
		cool = false;
		cudaSetDevice(0);
		m_size = size;
		m_ptrA = a;
		// allocate the needed memory on the gpu
		{
			checkStatus(cudaMalloc((void**)&devA, sizeof(T) * size), "cudaMalloc");
		}
		// copy data from cpu (host) memory to gpu (device) memory

		{
			checkStatus(cudaMemcpy(devA, m_ptrA, size * sizeof(T), cudaMemcpyHostToDevice),"cudaMemcpy of type host to device");
		}
	}

	virtual void coolDown() {
		cool = true;
		// copy memory from the gpu back to the cpu
		{
			checkStatus(cudaMemcpy(m_ptrA, devA, m_size * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy of type device to host");
		}
		// synchronize the device
		checkStatus(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
		release();
	}
};