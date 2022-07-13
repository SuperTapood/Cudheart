#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"

#include <stdio.h>
#include <iostream>
#include <chrono>
#include <ctime>
#include <device_functions.h>

std::chrono::duration<double> addWithCudaUni(int* c, const int* a, const int* b, unsigned int size);

std::chrono::duration<double> addWithCuda(int* c, const int* a, const int* b, unsigned int size);