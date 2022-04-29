#pragma once

#include "../../Arrays/Arrays.cuh"
#include <cmath>

using Cudheart::NDArrays::Vector;
using Cudheart::NDArrays::Matrix;
using namespace Cudheart::Exceptions;
using Cudheart::VectorOps::empty;
using Cudheart::VectorOps::emptyLike;

namespace Cudheart::CPP::Math::EMath {
	template <typename T>
	Vector<T>* squareRoot(Vector<T>* vec) {
		Vector<T>* output = emptyLike(vec);

		for (int i = 0; i < vec->getSize(); i++) {
			output->set(i, sqrt(vec->get(i)));
		}

        return output;
	}

	template <typename T>
	Matrix<T>* squareRoot(Matrix<T>* mat) {
		Vector<T>* flat = mat->flatten();

		Vector<T>* out = squareRoot(flat);

		delete flat;

		return fromVector(out, mat->getWidth(), mat->getHeight(), true);
	}

	template <typename T>
	Vector<T>* logarithm(Vector<T>* vec) {
		Vector<T>* output = emptyLike(vec);

		for (int i = 0; i < vec->getSize(); i++) {
			output->set(i, log(vec->get(i)));
		}

		return output;
	}

	template <typename T>
	Matrix<T>* logarithm(Matrix<T>* mat) {
		Vector<T>* flat = mat->flatten();

		Vector<T>* out = squareRoot(flat);

		delete flat;

		return fromVector(out, mat->getWidth(), mat->getHeight(), true);
	}
}