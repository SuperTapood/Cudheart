#pragma once

#include "../../Arrays/Arrays.cuh"
#include <cmath>

using Cudheart::NDArrays::Vector;
using Cudheart::NDArrays::Matrix;
using namespace Cudheart::Exceptions;
using Cudheart::VectorOps::empty;
using Cudheart::VectorOps::emptyLike;
using Cudheart::MatrixOps::fromVector;

namespace Cudheart::CPP::Math {
	template <typename T>
	NDArray<T>* squareRoot(NDArray<T>* arr) {
		NDArray<T>* output = arr->emptyLike();
		
		for (int i = 0; i < arr->getSize(); i++) {
			output->set(i, sqrt(arr->get(i)));
		}

		return output;
	}

	template <typename T>
	NDArray<T>* power(NDArray<T>* base, T po) {
		NDArray<T>* output = base->emptyLike();

		for (int i = 0; i < base->getSize(); i++) {
			output->set(i, pow(base->get(i), po));
		}

		return output;
	}

	template <typename T>
	NDArray<T>* power(T base, NDArray<T>* po) {
		NDArray<T>* output = po->emptyLike();

		for (int i = 0; i < po->getSize(); i++) {
			output->set(i, pow(base, po->get(i)));
		}

		return output;
	}

	template <typename T>
	NDArray<T>* power(NDArray<T>* base, NDArray<T>* po) {
		base->assertMatchShape(po);

		NDArray<T>* output = base->emptyLike();

		for (int i = 0; i < base->getSize(); i++) {
			output->set(i, pow(base->get(i), po->get(i)));
		}

		return output;
	}

	template <typename T>
	Matrix<T>* power(Matrix<T>* base, Vector<T>* po) {
		if (base->getHeight() != po->getSize()) {
			Cudheart::Exceptions::ShapeMismatchException(base->getHeight(), po->getSize()).raise();
		}
		Matrix<T>* out = emptyLike<T>(base);

		for (int i = 0; i < base->getHeight(); i++) {
			T p = po->get(i);
			for (int j = 0; j < base->getWidth(); j++) {
				out->set(i, j, pow(p, base->get(i, j)));
			}
		}

		return out;
	}

	template <typename T>
	NDArray<T>* around(NDArray<T>* arr, int decimals) {
		NDArray<T>* out = arr->emptyLike();
		int v = std::pow(10, decimals);

		for (int i = 0; i < out->getSize(); i++) {
			T value = arr->get(i) * v;
			value = std::round(value);
			out->set(i, value / v);
		}

		return out;
	}

	template <typename T>
	NDArray<T>* around(NDArray<T>* arr) {
		return around(arr, 0);
	}

	template <typename T>
	NDArray<T>* rint(NDArray<T>* arr) {
		NDArray<T>* out = emptyLike(arr);

		for (int i = 0; i < out->getSize(); i++) {
			out->set(i, std::rint(arr->get(i)));
		}

		return out;
	}

	template <typename T>
	NDArray<T>* fix(NDArray<T>* x) {
		NDArray<T>* out = x->emptyLike();

		for (int i = 0; i < x->getSize(); i++) {
			T v = x->get(i);
			if (v > 0) {
				out->set(i, std::floor(v));
			}
			else {
				out->set(i, std::ceil(v));
			}
		}

		return out;
	}

	template <typename T>
	NDArray<T>* floor(NDArray<T>* x) {
		NDArray<T>* out = x->emptyLike();

		for (int i = 0; i < x->getSize(); i++) {
			out->set(i, std::floor(x->get(i)));
		}

		return out;
	}

	template <typename T>
	NDArray<T>* ceil(NDArray<T>* x) {
		NDArray<T>* out = x->emptyLike();

		for (int i = 0; i < x->getSize(); i++) {
			out->set(i, std::ceil(x->get(i)));
		}

		return out;
	}

	template <typename T>
	NDArray<T>* trunc(NDArray<T>* x) {
		NDArray<T>* out = x->emptyLike();

		for (int i = 0; i < x->getSize(); i++) {
			out->set(i, std::trunc(x->get(i)));
		}

		return out;
	}

	template <typename T>
	T prod(NDArray<T>* x) {
		T out = (T)1;
		
		for (int i = 0; i < x->getSize(); i++) {
			out *= x->get(i);
		}
		
		return out;
	}

	template <typename T>
	T sum(NDArray<T>* x) {
		T out = (T)0;

		for (int i = 0; i < x->getSize(); i++) {
			out += x->get(i);
		}

		return out;
	}

	template <typename T>
	NDArray<T>* cumProd(NDArray<T>* x) {
		NDArray<T>* out = x->emptyLike();

		T prod = (T)1;
		
		for (int i = 0; i < x->getSize(); i++) {
			prod *= x->get(i);
			out->set(i, prod);
		}

		return out;
	}

	template <typename T>
	NDArray<T>* cumSum(NDArray<T>* x) {
		NDArray<T>* out = x->emptyLike();

		T prod = (T)0;

		for (int i = 0; i < x->getSize(); i++) {
			prod += x->get(i);
			out->set(i, prod);
		}

		return out;
	}
}