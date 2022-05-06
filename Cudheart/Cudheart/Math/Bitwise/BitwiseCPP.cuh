#pragma once

#include "../../Arrays/Arrays.cuh"

using Cudheart::NDArrays::Vector;
using Cudheart::NDArrays::Matrix;
using namespace Cudheart::Exceptions;
using Cudheart::VectorOps::empty;

namespace Cudheart::CPP::Math::Bitwise {
	template <typename T>
	Vector<T>* bitwiseAnd(Vector<T>* a, Vector<T>* b) {
		a->assertMatchSize(b);

		int len = a->getSize();
		Vector<T>* out = empty<T>(len);

		for (int i = 0; i < len; i++) {
			out->set(i, a->get(i) & b->get(i));
		}

		return out;
	}

	template <typename T>
	Matrix<T>* bitwiseAnd(Matrix<T>* a, Matrix<T>* b) {
		a->assertMatchSize(b);

		Vector<T>* flat = bitwiseAnd(a->flatten(), b->flatten());

		return fromVector(flat, a->getWidth(), b->getHeight(), true);
	}

	template <typename T>
	Vector<T>* bitwiseOr(Vector<T>* a, Vector<T>* b) {
		a->assertMatchSize(b);

		int len = a->getSize();
		Vector<T>* out = empty<T>(len);

		for (int i = 0; i < len; i++) {
			out->set(i, a->get(i) | b->get(i));
		}

		return out;
	}

	template <typename T>
	Matrix<T>* bitwiseOr(Matrix<T>* a, Matrix<T>* b) {
		a->assertMatchSize(b);

		Vector<T>* flat = bitwiseOr(a->flatten(), b->flatten());

		return fromVector(flat, a->getWidth(), b->getHeight(), true);
	}

	template <typename T>
	Vector<T>* bitwiseXor(Vector<T>* a, Vector<T>* b) {
		a->assertMatchSize(b);

		int len = a->getSize();
		Vector<T>* out = empty<T>(len);

		for (int i = 0; i < len; i++) {
			out->set(i, a->get(i) ^ b->get(i));
		}

		return out;
	}

	template <typename T>
	Matrix<T>* bitwiseXor(Matrix<T>* a, Matrix<T>* b) {
		a->assertMatchSize(b);

		Vector<T>* flat = bitwiseXor(a->flatten(), b->flatten());

		return fromVector(flat, a->getWidth(), b->getHeight(), true);
	}

	template <typename T>
	Vector<T>* bitwiseLeftShift(Vector<T>* a, Vector<T>* b) {
		a->assertMatchSize(b);

		int len = a->getSize();
		Vector<T>* out = empty<T>(len);

		for (int i = 0; i < len; i++) {
			out->set(i, a->get(i) << b->get(i));
		}

		return out;
	}

	template <typename T>
	Matrix<T>* bitwiseLeftShift(Matrix<T>* a, Matrix<T>* b) {
		a->assertMatchSize(b);

		Vector<T>* flat = bitwiseLeftShift(a->flatten(), b->flatten());

		return fromVector(flat, a->getWidth(), b->getHeight(), true);
	}

	template <typename T>
	Vector<T>* bitwiseRightShift(Vector<T>* a, Vector<T>* b) {
		a->assertMatchSize(b);

		int len = a->getSize();
		Vector<T>* out = empty<T>(len);

		for (int i = 0; i < len; i++) {
			out->set(i, a->get(i) >> b->get(i));
		}

		return out;
	}

	template <typename T>
	Matrix<T>* bitwiseRightShift(Matrix<T>* a, Matrix<T>* b) {
		a->assertMatchSize(b);

		Vector<T>* flat = bitwiseRightShift(a->flatten(), b->flatten());

		return fromVector(flat, a->getWidth(), b->getHeight(), true);
	}

	template <typename T>
	Vector<T>* bitwiseNot(Vector<T>* vec) {
		int len = vec->getSize();
		Vector<T>* out = empty<T>(len);

		for (int i = 0; i < len; i++) {
			out->set(i, ~(vec->get(i)));
		}

		return out;
	}

	template <typename T>
	Matrix<T>* bitwiseNot(Matrix<T>* mat) {
		Vector<T>* flat = bitwiseNot(mat->flatten());

		return fromVector(flat, mat->getWidth(), mat->getHeight(), true);
	}
}