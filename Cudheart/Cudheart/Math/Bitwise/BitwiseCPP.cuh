#pragma once

#include "../Arrays/Arrays.cuh"

using Cudheart::NDArrays::Vector;
using Cudheart::NDArrays::Matrix;
using namespace Cudheart::Exceptions;
using Cudheart::VectorOps::empty;


namespace Cudheart::Math::CPP::Bitwise {
	template <typename T>
	static Vector<T>* bitwiseAnd(Vector<T>* a, Vector<T>* b) {
		if (a->getSize() != b->getSize()) {
			throw new ShapeMismatchException(a->getSize(), b->getSize());
		}

		int len = a->getSize();
		Vector<T>* out = empty<T>(len);

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

	template <typename T>
	static Vector<T>* bitwiseOr(Vector<T>* a, Vector<T>* b) {
		if (a->getSize() != b->getSize()) {
			throw new ShapeMismatchException(a->getSize(), b->getSize());
		}

		int len = a->getSize();
		Vector<T>* out = empty<T>(len);

		for (int i = 0; i < len; i++) {
			out->set(i, a->get(i) | b->get(i));
		}

		return out;
	}

	template <typename T>
	static Matrix<T>* bitwiseOr(Matrix<T>* a, Matrix<T>* b) {
		if (a->getWidth() != b->getWidth() || a->getHeight() != b->getHeight()) {
			throw new ShapeMismatchException(a->getWidth(), a->getHeight(), b->getWidth(), b->getHeight());
		}

		Vector<T>* flat = bitwiseOr(a->flatten(), b->flatten());

		return fromVector(flat, a->getWidth(), b->getHeight());
	}

	template <typename T>
	static Vector<T>* bitwiseXor(Vector<T>* a, Vector<T>* b) {
		if (a->getSize() != b->getSize()) {
			throw new ShapeMismatchException(a->getSize(), b->getSize());
		}

		int len = a->getSize();
		Vector<T>* out = empty<T>(len);

		for (int i = 0; i < len; i++) {
			out->set(i, a->get(i) ^ b->get(i));
		}

		return out;
	}

	template <typename T>
	static Matrix<T>* bitwiseXor(Matrix<T>* a, Matrix<T>* b) {
		if (a->getWidth() != b->getWidth() || a->getHeight() != b->getHeight()) {
			throw new ShapeMismatchException(a->getWidth(), a->getHeight(), b->getWidth(), b->getHeight());
		}

		Vector<T>* flat = bitwiseXor(a->flatten(), b->flatten());

		return fromVector(flat, a->getWidth(), b->getHeight());
	}

	template <typename T>
	static Vector<T>* bitwiseLeftShift(Vector<T>* a, Vector<T>* b) {
		if (a->getSize() != b->getSize()) {
			throw new ShapeMismatchException(a->getSize(), b->getSize());
		}

		int len = a->getSize();
		Vector<T>* out = empty<T>(len);

		for (int i = 0; i < len; i++) {
			out->set(i, a->get(i) << b->get(i));
		}

		return out;
	}

	template <typename T>
	static Matrix<T>* bitwiseLeftShift(Matrix<T>* a, Matrix<T>* b) {
		if (a->getWidth() != b->getWidth() || a->getHeight() != b->getHeight()) {
			throw new ShapeMismatchException(a->getWidth(), a->getHeight(), b->getWidth(), b->getHeight());
		}

		Vector<T>* flat = bitwiseLeftShift(a->flatten(), b->flatten());

		return fromVector(flat, a->getWidth(), b->getHeight());
	}

	template <typename T>
	static Vector<T>* bitwiseRightShift(Vector<T>* a, Vector<T>* b) {
		if (a->getSize() != b->getSize()) {
			throw new ShapeMismatchException(a->getSize(), b->getSize());
		}

		int len = a->getSize();
		Vector<T>* out = empty<T>(len);

		for (int i = 0; i < len; i++) {
			out->set(i, a->get(i) >> b->get(i));
		}

		return out;
	}

	template <typename T>
	static Matrix<T>* bitwiseRightShift(Matrix<T>* a, Matrix<T>* b) {
		if (a->getWidth() != b->getWidth() || a->getHeight() != b->getHeight()) {
			throw new ShapeMismatchException(a->getWidth(), a->getHeight(), b->getWidth(), b->getHeight());
		}

		Vector<T>* flat = bitwiseRightShift(a->flatten(), b->flatten());

		return fromVector(flat, a->getWidth(), b->getHeight());
	}

	template <typename T>
	static Vector<T>* bitwiseNot(Vector<T>* vec) {
		int len = vec->getSize();
		Vector<T>* out = empty<T>(len);

		for (int i = 0; i < len; i++) {
			out->set(i, ~(vec->get(i)));
		}

		return out;
	}

	template <typename T>
	static Matrix<T>* bitwiseNot(Matrix<T>* mat) {
		Vector<T>* flat = bitwiseNot(mat->flatten());

		return fromVector(flat, mat->getWidth(), mat->getHeight());
	}
}