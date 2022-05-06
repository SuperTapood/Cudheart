#pragma once

#include "BaseException.cuh"
#include "driver_types.h"
#include <iostream>

using namespace std;

namespace Cudheart::Exceptions {
	class CudaException : public BaseException {
	public:
		CudaException(cudaError_t error, string func) {
			// add more exceptions as i encounter them
			ostringstream os;
			os << "CudaException in '" << func << "': ";
			switch (error) {
			case (cudaErrorInvalidValue):
				os << "function got passed a value outside of an acceptable range";
				break;
			case (cudaErrorMemoryAllocation):
				os << "cuda could not allocate enough memory on the GPU's VRAM";
				break;
			case (cudaErrorInitializationError):
				os << "cuda could not initialize the cuda driver and runtime";
			default:
				os << "unknown error: '" << cudaGetErrorString(error) << "'";
			}
			m_msg = os.str();
		}
	};
}