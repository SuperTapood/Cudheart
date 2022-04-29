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
	Vector<T>* loga(Vector<T>* vec) {
		Vector<T>* output = emptyLike(vec);

		for (int i = 0; i < vec->getSize(); i++) {
			output->set(i, log(vec->get(i)));
		}

		return output;
	}

	template <typename T>
	Matrix<T>* loga(Matrix<T>* mat) {
		Vector<T>* flat = mat->flatten();

		Vector<T>* out = loga(flat);

		delete flat;

		return fromVector(out, mat->getWidth(), mat->getHeight(), true);
	}

	template <typename T>
	Vector<T>* loga2(Vector<T>* vec) {
		Vector<T>* output = emptyLike(vec);

		for (int i = 0; i < vec->getSize(); i++) {
			output->set(i, log2(vec->get(i)));
		}

		return output;
	}

	template <typename T>
	Matrix<T>* loga2(Matrix<T>* mat) {
		Vector<T>* flat = mat->flatten();

		Vector<T>* out = loga2(flat);

		delete flat;

		return fromVector(out, mat->getWidth(), mat->getHeight(), true);
	}

	template <typename T>
	Vector<T>* logan(Vector<T>* vec, T n) {
		Vector<T>* output = emptyLike(vec);

		for (int i = 0; i < vec->getSize(); i++) {
			// using the change of bases rule:
			// log n of vec[i] = log(vec[i]) / log(n)
			output->set(i, (log(vec->get(i)) / log(n)));
		}

		return output;
	}

	template <typename T>
	Matrix<T>* logan(Matrix<T>* mat, T n) {
		Vector<T>* flat = mat->flatten();

		Vector<T>* out = logan(flat, n);

		delete flat;

		return fromVector(out, mat->getWidth(), mat->getHeight(), true);
	}

	template <typename T>
	Vector<T>* loga10(Vector<T>* vec) {
		Vector<T>* output = emptyLike(vec);

		for (int i = 0; i < vec->getSize(); i++) {
			output->set(i, log10(vec->get(i)));
		}

		return output;
	}

	template <typename T>
	Matrix<T>* loga10(Matrix<T>* mat) {
		Vector<T>* flat = mat->flatten();

		Vector<T>* out = loga10(flat);

		delete flat;

		return fromVector(out, mat->getWidth(), mat->getHeight(), true);
	}
}