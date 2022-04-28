#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


template <typename T>
class ContainerA {
private:
	void** m_ptrA;
	cudaError_t cudaStatus;
	int m_size;

public:
	T* devA;

private:
	void error() {
		fprintf(stderr, "cudaMalloc failed!");
		release();

	}

	void release() {
		cudaFree(m_ptrA);
	}

	void allocate(T* ptr, int size) {
		cudaStatus = cudaMalloc(ptr, sizeof(T) * size);
		if (cudaStatus != cudaSuccess) {
			error();
		}
	}

	void copyMemHTD(T* dst, T* src, int size) {
		cudaStatus = cudaMemcpy(dst, src, size * sizeof(T), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			error();
		}
	}

	void copyMemDTH(T* dst, T* src, int size) {
		cudaStatus = cudaMemcpy(dst, src, size * sizeof(T), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			error();
		}
	}

public:
	virtual void warmUp(void** a, int size) {
		m_size = size;
		m_ptrA = a;
		allocate(devA, size);
		copyMemHTD(devA, m_ptrA, size);
	}

	virtual void coolDown() {
		copyMemDTH(m_ptrA, devA, m_size);
		release();
	}
};