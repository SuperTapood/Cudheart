#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


namespace Cudheart::Kernels::Math::Bitwise {
    template <typename T>
    __global__ void kernelBitwiseAnd(T* c, const T* a, const T* b)
    {
        int i = threadIdx.x;
        c[i] = a[i] & b[i];
    }

    template <typename T>
    __global__ void kernelBitwiseOr(T* c, const T* a, const T* b)
    {
        int i = threadIdx.x;
        c[i] = a[i] | b[i];
    }

    template <typename T>
    __global__ void kernelBitwiseXor(T* c, const T* a, const T* b)
    {
        int i = threadIdx.x;
        c[i] = a[i] ^ b[i];
    }

    template <typename T>
    __global__ void kernelBitwiseLeftShift(T* c, const T* a, const T* b)
    {
        int i = threadIdx.x;
        c[i] = a[i] << b[i];
    }

    template <typename T>
    __global__ void kernelBitwiseRightShift(T* c, const T* a, const T* b)
    {
        int i = threadIdx.x;
        c[i] = a[i] >> b[i];
    }

    template <typename T>
    __global__ void kernelBitwiseNot(T* output, const T* input)
    {
        int i = threadIdx.x;
        output[i] = ~input[i];
    }
}