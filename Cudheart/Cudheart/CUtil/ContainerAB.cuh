#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

template <typename T>
class ContainerAB {
private:
	void** m_ptrA;
	void** m_ptrB;
	cudaError_t cudaStatus;
	int m_size;

public:
	T* devA;
	T* devB;

private:
	void error() {
		fprintf(stderr, "cudaMalloc failed!");
		release();
	}

	void release() {
		cudaFree(m_ptrA);
		cudaFree(m_ptrB);
	}

	void allocate(T* ptr, int size) {
		cudaStatus = cudaMalloc((void**)&ptr, sizeof(T) * size);
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
	virtual void warmUp(void** a, void** b, int size) {
		m_size = size;
		m_ptrA = a;
		m_ptrB = b;
		allocate(devA, size);
		allocate(devB, size);
		copyMemHTD(devA, m_ptrA, size);
		copyMemHTD(devB, m_ptrB, size);
	}

	virtual void coolDown() {
		copyMemDTH(m_ptrA, devA, m_size);
		copyMemDTH(m_ptrB, devB, m_size);
		release();
	}
};