#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


template <typename T>
__global__ void kernelBitwiseAnd(T* c, const T* a, const T* b)
{
    int i = threadIdx.x;
    c[i] = a[i] & b[i];
}