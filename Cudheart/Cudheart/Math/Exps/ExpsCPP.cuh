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
}