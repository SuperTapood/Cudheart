#pragma once

#include "../../Arrays/Arrays.cuh"
#include <cmath>
#include "../../Logic/Logic.cuh"

using Cudheart::NDArrays::Vector;
using Cudheart::NDArrays::Matrix;
using Cudheart::NDArrays::NDArray;
using namespace Cudheart::Exceptions;
using Cudheart::VectorOps::empty;
using Cudheart::VectorOps::emptyLike;
using Cudheart::MatrixOps::fromVector;

namespace Cudheart::Math::CPP::Statistics {
	template <typename T>
	int ptp(NDArray<T>* a) {
		return Cudheart::Logic::amax(a) - Cudheart::Logic::amin(a);
	}
	
	
	
}