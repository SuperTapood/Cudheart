#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_runtime_api.h"


namespace Cudheart::Kernels::Math::EMath {
	template <typename T>
	__global__ void kernelArccos(T* out, const T* in) 
	{
		int i = threadIdx.x;
		out[i] = acos(in[i]);
	}

	template <typename T>
	__global__ void kernelArcsin(T* out, const T* in)
	{
		int i = threadIdx.x;
		out[i] = asin(in[i]);
	}

	template <typename T>
	__global__ void kernelArctan(T* out, const T* in)
	{
		int i = threadIdx.x;
		out[i] = atan(in[i]);
	}

	template <typename T>
	__global__ void kernelArccot(T* out, const T* in)
	{
		int i = threadIdx.x;
		long double pi = 3.1415926535897932384626433;
		out[i] = (pi / 2) - atan(in[i]));
	}

	template <typename T>
	__global__ void kernelLog(T* out, const T* in)
	{
		int i = threadIdx.x;
		out[i] = log(in[i]);
	}

	template <typename T>
	__global__ void kernelLog10(T* out, const T* in)
	{
		int i = threadIdx.x;
		out[i] = log10(in[i]);
	}

	template <typename T>
	__global__ void kernelLog2(T* out, const T* in)
	{
		int i = threadIdx.x;
		out[i] = log2(in[i]);
	}

	template <typename T>
	__global__ void kernelLogN(T* out, const T* in, T n)
	{
		int i = threadIdx.x;
		out[i] = (log(in[i]) / log(n));
	}

	template <typename T>
	__global__ void kernelPower(T* c, const T* a, const T* b)
	{
		int i = threadIdx.x;
		c[i] = pow(a[i], b[i]);
	}

	template <typename T>
	__global__ void kernelSqrt(T* out, const T* in)
	{
		int i = threadIdx.x;
		out[i] = sqrt(in[i]);
	}
}