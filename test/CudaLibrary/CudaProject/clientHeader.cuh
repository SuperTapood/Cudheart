#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "float.h"

import ModuleA;

cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);

template <typename T>
T add(T a, T b);