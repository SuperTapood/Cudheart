#pragma once

#include "../../Arrays/Arrays.cuh"
#include <cmath>
#include "../Constants.cuh"

using Cudheart::NDArrays::Vector;
using Cudheart::NDArrays::Matrix;
using namespace Cudheart::Exceptions;
using Cudheart::VectorOps::empty;
using Cudheart::VectorOps::emptyLike;
using Cudheart::MatrixOps::fromVector;

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

	template <typename T>
	Vector<T>* power(Vector<T>* base, T po) {
		Vector<T>* out = emptyLike(base);

		for (int i = 0; i < out->getSize(); i++) {
			out->set(i, pow(base->get(i), power));
		}

		return out;
	}

	template <typename T>
	Vector<T>* power(Vector<T>* base, Vector<T>* po) {
		base->assertMatchSize(power);
		Vector<T>* out = emptyLike(base);

		for (int i = 0; i < out->getSize(); i++) {
			out->set(i, pow(base->get(i), po->get(i)));
		}

		return out;
	}

	template <typename T>
	Matrix<T>* power(Matrix<T>* base, T po) {
		Vector<T>* flat = base->flatten();

		Vector<T>* out = power(flat, power);

		delete flat;

		return fromVector(out, base->getWidth(), base->getHeight());
	}

	template <typename T>
	Matrix<T>* power(Matrix<T>* base, Matrix<T>* high) {
		base->assertMatchSize(power);
		Vector<T>* b = base->flatten();
		Vector<T>* p = high->flatten();

		Vector<T>* output = power(b, p);

		delete b;
		delete p;

		return fromVector(output, base->getWidth(), base->getHeight());
	}

	template <typename T>
	Matrix<T>* power(Matrix<T>* base, Vector<T>* po) {
		if (base->getHeight() != po->getSize()) {
			Cudheart::Exceptions::ShapeMismatchException(base->getHeight(), po->getSize()).raise();
		}
		Matrix<T>* out = emptyLike(base);

		for (int i = 0; i < base->getHeight(); i++) {
			T p = po->get(i);
			for (int j = 0; j < base->getWidth(); j++) {
				out->set(i, j, pow(p, base->get(i, j)));
			}
		}

		return out;
	}

	template <typename T>
	Vector<T>* arccos(Vector<T>* vec) {
		Vector<T>* out = emptyLike(vec);
		for (int i = 0; i < vec->getSize(); i++) {
			out->set(i, acos(vec->get(i)));
		}
		return out;
	}

	template <typename T>
	Matrix<T>* arccos(Matrix<T>* mat) {
		Vector<T>* flat = mat->flatten();

		Vector<T>* out = arccos(flat);

		delete flat;

		return fromVector(out, mat->getWidth(), mat->getHeight(), true);
	}

	template <typename T>
	Vector<T>* arcsin(Vector<T>* vec) {
		Vector<T>* out = emptyLike(vec);
		for (int i = 0; i < vec->getSize(); i++) {
			out->set(i, asin(vec->get(i)));
		}
		return out;
	}

	template <typename T>
	Matrix<T>* arcsin(Matrix<T>* mat) {
		Vector<T>* flat = mat->flatten();

		Vector<T>* out = arccos(flat);

		delete flat;

		return fromVector(out, mat->getWidth(), mat->getHeight(), true);
	}

	template <typename T>
	Vector<T>* arctan(Vector<T>* vec) {
		Vector<T>* out = emptyLike(vec);
		for (int i = 0; i < vec->getSize(); i++) {
			out->set(i, atan(vec->get(i)));
		}
		return out;
	}

	template <typename T>
	Matrix<T>* arctan(Matrix<T>* mat) {
		Vector<T>* flat = mat->flatten();

		Vector<T>* out = arctan(flat);

		delete flat;

		return fromVector(out, mat->getWidth(), mat->getHeight(), true);
	}

	template <typename T>
	Vector<T>* arccot(Vector<T>* vec) {
		Vector<T>* out = emptyLike(vec);
		long double pi = 3.1415926535897932384626433;
		for (int i = 0; i < vec->getSize(); i++) {
			out->set(i, (pi / 2) - atan(vec->get(i)));
		}
		return out;
	}

	template <typename T>
	Matrix<T>* arccot(Matrix<T>* mat) {
		Vector<T>* flat = mat->flatten();

		Vector<T>* out = arccot(flat);

		delete flat;

		return fromVector(out, mat->getWidth(), mat->getHeight(), true);
	}
}