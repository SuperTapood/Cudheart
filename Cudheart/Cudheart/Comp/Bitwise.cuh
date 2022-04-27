#pragma once

#include "../Arrays/Arrays.cuh"


namespace Cudheart::Comp::Bitwise {
	using namespace Cudheart::NDArrays;
	using namespace Cudheart::MatrixOps;
	using namespace Cudheart::VectorOps;
	using namespace Cudheart::Exceptions;

	template <typename T>
	static Vector<T>* bitwiseAnd(Vector<T>* a, Vector<T>* b) {
		if (a->getSize() != b->getSize()) {
			throw new ShapeMismatchException(a->getSize(), b->getSize());
		}

		int len = a->getSize();
		Vector<T>* out = empty(len);

		for (int i = 0; i < len; i++) {
			out->set(i, a->get(i) & b->get(i));
		}

		return out;
	}

	template <typename T>
	static Matrix<T>* bitwiseAnd(Matrix<T>* a, Matrix<T>* b) {
		if (a->getWidth() != b->getWidth() || a->getHeight() != b->getHeight()) {
			throw new ShapeMismatchException(a->getWidth(), a->getHeight(), b->getWidth(), b->getHeight());
		}

		Vector<T>* flat = bitwiseAnd(a->flatten(), b->flatten());

		return fromVector(flat, a->getWidth(), b->getHeight());
	}
}
