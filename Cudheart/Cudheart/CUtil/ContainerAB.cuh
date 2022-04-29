#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../Exceptions/CudaException.cuh"

#include <stdio.h>

template <typename T>
class ContainerABC {
private:
	T* m_ptrA;
	T* m_ptrB;
	cudaError_t cudaStatus;
	int m_size;

public:
	T* devA;
	T* devB;

private:
	void release() {
		cudaFree((void*)devA);
		cudaFree((void*)devB);
	}

	void checkStatus(cudaError_t status, string func) {
		if (status != cudaSuccess) {
			release();
			Cudheart::Exceptions::CudaException(status, func).raise();
		}
	}

public:
	virtual void warmUp(T* a, T* b, T* c, int size) {
		cudaSetDevice(0);
		m_size = size;
		m_ptrA = a;
		m_ptrB = b;
		// allocate the needed memory on the gpu
		{
			checkStatus(cudaMalloc((void**)&devA, sizeof(T) * size), "cudaMalloc");
			checkStatus(cudaMalloc((void**)&devB, sizeof(T) * size), "cudaMalloc");
		}
		// copy data from cpu (host) memory to gpu (device) memory

		{
			checkStatus(cudaMemcpy(devA, m_ptrA, size * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy of type host to device");
			checkStatus(cudaMemcpy(devB, m_ptrB, size * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy of type host to device");
		}
	}

	virtual void coolDown() {
		// copy memory from the gpu back to the cpu
		{
			checkStatus(cudaMemcpy(m_ptrA, devA, m_size * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy of type device to host");
			checkStatus(cudaMemcpy(m_ptrB, devB, m_size * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy of type device to host");
		}
		// synchronize the device
		checkStatus(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
		release();
	}
};