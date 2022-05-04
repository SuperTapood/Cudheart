#pragma once

#include "../Arrays/Arrays.cuh"

namespace Cudheart::Logic {
	template <typename T>
	bool all(Vector<T>* vec) {
		for (int i = 0; i < vec->getSize(); i++) {
			if (!(bool)vec->get(i)) {
				return false;
			}
		}
		
		return true;
	}

	template <typename T>
	bool all(Matrix<T>* mat) {
		for (int i = 0; i < mat->getSize(); i++) {
			if (!(bool)mat->get(i)) {
				return false;
			}
		}

		return true;
	}

	template <typename T>
	bool any(Vector<T>* vec) {
		for (int i = 0; i < vec->getSize(); i++) {
			if ((bool)vec->get(i)) {
				return true;
			}
		}

		return false;
	}

	template <typename T>
	bool any(Matrix<T>* mat) {
		for (int i = 0; i < mat->getSize(); i++) {
			if ((bool)mat->get(i)) {
				return true;
			}
		}

		return false;
	}

	template <typename T>
	Vector<bool>* logicalAnd(Vector<T>* a, Vector<T>* b) {
		a->assertMatchSize(b);
		
		Vector<bool>* out = new Vector<bool>(a->getSize());

		for (int i = 0; i < a->getSize(); i++) {
			out->set(i, a->get(i) && b->get(i));
		}

		return out;
	}

	template <typename T>
	Matrix<bool>* logicalAnd(Matrix<T>* a, Matrix<T>* b) {
		a->assertMatchSize(b);

		Matrix<bool>* out = new Matrix<bool>(a->getSize());

		for (int i = 0; i < a->getSize(); i++) {
			out->set(i, a->get(i) && b->get(i));
		}

		return out;
	}

	template <typename T>
	Matrix<bool>* logicalAnd(Matrix<T>* mat, Vector<T>* vec) {
		// assert mat width == vec size
		
		Matrix<bool>* out = new Matrix<bool>(mat->getHeight(), mat->getWidth());
		
		for (int i = 0; i < mat->getHeight(); i++) {
			for (int j = 0; j < mat->getWidth(); j++) {
				out->set(i, j, mat->get(i, j) && vec->get(j));
			}
		}
		
		return out;
	}

	template <typename T> 
	Matrix<bool>* logicalAnd(Vector<T>* vec, Matrix<T>* mat) {
		return logicalAnd(mat, vec);
	}

	template <typename T>
	Vector<bool>* logicalOr(Vector<T>* a, Vector<T>* b) {
		a->assertMatchSize(b);

		Vector<bool>* out = new Vector<bool>(a->getSize());

		for (int i = 0; i < a->getSize(); i++) {
			out->set(i, a->get(i) || b->get(i));
		}

		return out;
	}

	template <typename T>
	Matrix<bool>* logicalOr(Matrix<T>* a, Matrix<T>* b) {
		a->assertMatchSize(b);

		Matrix<bool>* out = new Matrix<bool>(a->getSize());

		for (int i = 0; i < a->getSize(); i++) {
			out->set(i, a->get(i) || b->get(i));
		}

		return out;
	}

	template <typename T>
	Matrix<bool>* logicalOr(Matrix<T>* mat, Vector<T>* vec) {
		// assert mat width == vec size

		Matrix<bool>* out = new Matrix<bool>(mat->getHeight(), mat->getWidth());

		for (int i = 0; i < mat->getHeight(); i++) {
			for (int j = 0; j < mat->getWidth(); j++) {
				out->set(i, j, mat->get(i, j) || vec->get(j));
			}
		}

		return out;
	}

	template <typename T>
	Matrix<bool>* logicalOr(Vector<T>* vec, Matrix<T>* mat) {
		return logicalOr(mat, vec);
	}
}