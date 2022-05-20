#pragma once

#include "../../Arrays/Arrays.cuh"
#include <cmath>

using Cudheart::NDArrays::Vector;
using Cudheart::NDArrays::Matrix;
using namespace Cudheart::Exceptions;
using Cudheart::VectorOps::empty;
using Cudheart::VectorOps::emptyLike;
using Cudheart::MatrixOps::fromVector;

namespace Cudheart::CPP::Math::Exp {
	template <typename T>
	NDArray<T>* loga(NDArray<T>* x) {
		NDArray<T>* output = x->emptyLike();

		for (int i = 0; i < x->getSize(); i++) {
			output->set(i, log(x->get(i)));
		}

		return output;
	}

	template <typename T>
	NDArray<T>* loga2(NDArray<T>* x) {
		NDArray<T>* output = x->emptyLike();

		for (int i = 0; i < x->getSize(); i++) {
			output->set(i, log2(x->get(i)));
		}

		return output;
	}

	template <typename T>
	NDArray<T>* logan(NDArray<T>* x, T n) {
		NDArray<T>* output = x->emptyLike();

		for (int i = 0; i < x->getSize(); i++) {
			// using the change of bases rule:
			// log n of x[i] = log(x[i]) / log(n)
			output->set(i, (log(x->get(i)) / log(n)));
		}

		return output;
	}

	template <typename T>
	NDArray<T>* loga10(NDArray<T>* x) {
		NDArray<T>* output = x->emptyLike();

		for (int i = 0; i < x->getSize(); i++) {
			output->set(i, log10(x->get(i)));
		}

		return output;
	}

	template <typename T>
	NDArray<T>* expo(NDArray<T>* x) {
		NDArray<T>* output = x->emptyLike();

		for (int i = 0; i < x->getSize(); i++) {
			output->set(i, std::exp(x->get(i)));
		}

		return output;
	}
	
	template <typename T>
	NDArray<T>* expom1(NDArray<T>* x) {
		NDArray<T>* output = x->emptyLike();

		for (int i = 0; i < x->getSize(); i++) {
			output->set(i, std::expm1(x->get(i)));
		}

		return output;
	}

	template <typename T>
	NDArray<T>* expo2(NDArray<T>* x) {
		NDArray<T>* output = x->emptyLike();

		for (int i = 0; i < x->getSize(); i++) {
			output->set(i, std::exp2(x->get(i)));
		}

		return output;
	}

	template <typename T>
	NDArray<T>* logaddexp(NDArray<T>* a, NDArray<T>* b) {
		a->assertMatchShape(b);
		NDArray<T>* output = a->emptyLike();

		for (int i = 0; i < a->getSize(); i++) {
			output->set(i, std::log(std::exp(a->get(i)) + std::exp(b->get(i))));
		}

		return output;
	}

	template <typename T>
	NDArray<T>* logaddexp2(NDArray<T>* a, NDArray<T>* b) {
		a->assertMatchShape(b);
		NDArray<T>* output = a->emptyLike();

		for (int i = 0; i < a->getSize(); i++) {
			output->set(i, std::log(std::exp2(a->get(i)) + std::exp2(b->get(i))));
		}

		return output;
	}
}