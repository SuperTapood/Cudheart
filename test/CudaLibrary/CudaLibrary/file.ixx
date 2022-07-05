module moduleA;
export module ModuleA;

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "float.h"


cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);

template <typename T>
T add(T a, T b) {
	return a + b;
}