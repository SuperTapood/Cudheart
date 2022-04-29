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
	T* m_ptrC;
	cudaError_t cudaStatus;
	int m_size;

public:
	T* devA;
	T* devB;
	T* devC;

private:
	void release() {
		cudaFree((void*)devA);
		cudaFree((void*)devB);
		cudaFree((void*)devC);
	}

	void checkStatus(cudaError_t status) {
		if (status != cudaSuccess) {
			release();
			Cudheart::Exceptions::CudaException(cudaGetErrorName(status), cudaGetErrorString(status)).raise();
		}
	}

public:
	virtual void warmUp(T* a, T* b, T* c, int size) {
		cudaSetDevice(0);
		m_size = size;
		m_ptrA = a;
		m_ptrB = b;
		m_ptrC = c;
		checkStatus(cudaErrorInvalidValue);
		// allocate the needed memory on the gpu
		{
			checkStatus(cudaMalloc((void**)&devA, sizeof(T) * size));
			checkStatus(cudaMalloc((void**)&devB, sizeof(T) * size));
			checkStatus(cudaMalloc((void**)&devC, sizeof(T) * size));
		}
		// copy data from cpu (host) memory to gpu (device) memory

		{
			checkStatus(cudaMemcpy(devA, m_ptrA, size * sizeof(T), cudaMemcpyHostToDevice));
			checkStatus(cudaMemcpy(devB, m_ptrB, size * sizeof(T), cudaMemcpyHostToDevice));
			checkStatus(cudaMemcpy(devC, m_ptrB, size * sizeof(T), cudaMemcpyHostToDevice));
		}
	}

	virtual void coolDown() {
		// copy memory from the gpu back to the cpu
		{
			checkStatus(cudaMemcpy(m_ptrA, devA, m_size * sizeof(T), cudaMemcpyDeviceToHost));
			checkStatus(cudaMemcpy(m_ptrB, devB, m_size * sizeof(T), cudaMemcpyDeviceToHost));
			checkStatus(cudaMemcpy(m_ptrC, devC, m_size * sizeof(T), cudaMemcpyDeviceToHost));
		}
		// synchronize the device
		checkStatus(cudaDeviceSynchronize());		
		release();
	}
};