#pragma once

#include "../Util.cuh"
#include "../Arrays/Matrix.cuh"
#include "../Arrays/Vector.cuh"
#include "../Arrays/VectorOps.cuh"
#include "../Exceptions/Exceptions.cuh"
#include "StringType.cuh"

using Cudheart::NDArrays::Matrix;
using Cudheart::NDArrays::Vector;
using Cudheart::NDArrays::NDArray;
using Cudheart::VectorOps::zeros;
using namespace Cudheart::Exceptions;


namespace Cudheart::StringOps {
	inline NDArray<StringType*>* add(NDArray<StringType*>* x1, NDArray<StringType*>* x2) {
		x1->assertMatchShape(x2->getShape());

		NDArray<StringType*>* out = x1->emptyLike();

		for (int i = 0; i < x1->getSize(); i++) {
			out->set(i, new StringType(x1->get(i)->m_str + x2->get(i)->m_str));
		}

		return out;
	}
}