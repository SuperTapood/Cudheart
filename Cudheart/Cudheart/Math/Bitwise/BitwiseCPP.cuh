#pragma once

#include "../../Arrays/Arrays.cuh"

using Cudheart::NDArrays::Vector;
using Cudheart::NDArrays::Matrix;
using namespace Cudheart::Exceptions;
using Cudheart::VectorOps::empty;

namespace Cudheart::CPP::Math::Bitwise {
	template <typename T>
	NDArray<T>* bitwiseAnd(NDArray<T>* a, NDArray<T>* b) {
		a->AssertMatchShape(b);
		
		NDArray<T>* out = a->emptyLike();

		for (int i = 0; i < a->getSize(); i++) {
			out->set(i, a->get(i) & b->get(i));
		}

		return out;
	}

	template <typename T>
	NDArray<T>* bitwiseOr(NDArray<T>* a, NDArray<T>* b) {
		a->AssertMatchShape(b);

		NDArray<T>* out = a->emptyLike();

		for (int i = 0; i < a->getSize(); i++) {
			out->set(i, a->get(i) | b->get(i));
		}

		return out;
	}

	template <typename T>
	NDArray<T>* bitwiseXor(NDArray<T>* a, NDArray<T>* b) {
		a->AssertMatchShape(b);

		NDArray<T>* out = a->emptyLike();

		for (int i = 0; i < a->getSize(); i++) {
			out->set(i, a->get(i) ^ b->get(i));
		}

		return out;
	}

	template <typename T>
	NDArray<T>* bitwiseLeftShift(NDArray<T>* a, NDArray<T>* b) {
		a->AssertMatchShape(b);

		NDArray<T>* out = a->emptyLike();

		for (int i = 0; i < a->getSize(); i++) {
			out->set(i, a->get(i) << b->get(i));
		}

		return out;
	}

	template <typename T>
	NDArray<T>* bitwiseRightShift(NDArray<T>* a, NDArray<T>* b) {
		a->AssertMatchShape(b);

		NDArray<T>* out = a->emptyLike();

		for (int i = 0; i < a->getSize(); i++) {
			out->set(i, a->get(i) >> b->get(i));
		}

		return out;
	}

	template <typename T>
	NDArray<T>* bitwiseNot(NDArray<T>* x) {
		int len = x->getSize();
		NDArray<T>* out = x->emptyLike();

		for (int i = 0; i < len; i++) {
			out->set(i, ~(x->get(i)));
		}

		return out;
	}
}