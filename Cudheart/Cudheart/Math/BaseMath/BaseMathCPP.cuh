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
	Vector<T>* power(Vector<T>* base, T po) {
		Vector<T>* out = emptyLike(base);

		for (int i = 0; i < out->getSize(); i++) {
			out->set(i, pow(base->get(i), po));
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

		Vector<T>* out = power(flat, po);

		delete flat;

		return fromVector(out, base->getWidth(), base->getHeight());
	}

	template <typename T>
	Matrix<T>* power(Matrix<T>* base, Matrix<T>* high) {
		base->assertMatchSize(high);
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
	Vector<T>* around(Vector<T>* vec, int decimals) {
		Vector<T>* out = emptyLike(vec);
		int v = std::pow(10, decimals);
		
		for (int i = 0; i < out->getSize(); i++) {
			T value = vec->get(i) * v;
			value = std::round(value);
			out->set(i, value / v);
		}

		return out;
	}

	template <typename T>
	Vector<T>* around(Vector<T>* vec) {
		return around(vec, 0);
	}

	template <typename T>
	Matrix<T>* around(Matrix<T>* mat, int decimals) {
		Matrix<T>* out = emptyLike(vecmat;
		int v = std::pow(10, decimals);

		for (int i = 0; i < out->getSize(); i++) {
			T value = mat->get(i) * v;
			value = std::round(value);
			out->set(i, value / v);
		}

		return out;
	}

	template <typename T>
	Matrix<T>* around(Matrix<T>* mat) {
		return around(mat, 0);
	}
}