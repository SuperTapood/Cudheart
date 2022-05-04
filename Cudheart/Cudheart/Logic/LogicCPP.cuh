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

	template <typename T>
	Vector<bool>* logicalNot(Vector<T>* vec) {
		Vector<bool>* out = emptyLike(vec);

		for (int i = 0; i < vec->getSize(); i++) {
			out->set(i, !vec->get(i));
		}

		return out;
	}

	template <typename T>
	Matrix<bool>* logicalNot(Matrix<T>* mat) {
		Matrix<bool>* out = emptyLike(mat);

		for (int i = 0; i < mat->getSize(); i++) {
			out->set(i, !mat->get(i));
		}

		return out;
	}

	template <typename T>
	Vector<bool>* logicalXor(Vector<T>* a, Vector<T>* b) {
		a->assertMatchSize(b);

		Vector<bool>* out = new Vector<bool>(a->getSize());

		for (int i = 0; i < a->getSize(); i++) {
			out->set(i, a->get(i) ^ b->get(i));
		}

		return out;
	}

	template <typename T>
	Matrix<bool>* logicalXor(Matrix<T>* a, Matrix<T>* b) {
		a->assertMatchSize(b);

		Matrix<bool>* out = new Matrix<bool>(a->getSize());

		for (int i = 0; i < a->getSize(); i++) {
			out->set(i, a->get(i) ^ b->get(i));
		}

		return out;
	}

	template <typename T>
	Matrix<bool>* logicalXor(Matrix<T>* mat, Vector<T>* vec) {
		// assert mat width == vec size

		Matrix<bool>* out = new Matrix<bool>(mat->getHeight(), mat->getWidth());

		for (int i = 0; i < mat->getHeight(); i++) {
			for (int j = 0; j < mat->getWidth(); j++) {
				out->set(i, j, mat->get(i, j) ^ vec->get(j));
			}
		}

		return out;
	}

	template <typename T>
	Matrix<bool>* logicalXor(Vector<T>* vec, Matrix<T>* mat) {
		return logicalXor(mat, vec);
	}

	template <typename T>
	bool allclose(Vector<T>* a, Vector<T>* b, double rtol, double atol) {
		a->assertMatchSize(b);

		for (int i = 0; i < a->getSize(); i++) {
			T va = a->get(i);
			T vb = b->get(i);

			if (abs(va - vb) > (atol + rtol * abs(vb))) {
				return false;
			}
		}

		return true;
	}

	template <typename T>
	bool allclose(Vector<T>* a, Vector<T>* b, double rtol) {
		return allclose(a, b, rtol, 1e-08);
	}

	template <typename T>
	bool allclose(Vector<T>* a, Vector<T>* b) {
		return allclose(a, b, 1e-05, 1e-08);
	}

	template <typename T>
	bool allclose(Matrix<T>* a, Matrix<T>* b, double rtol, double atol) {
		a->assertMatchSize(b);

		for (int i = 0; i < a->getSize(); i++) {
			T va = a->get(i);
			T vb = b->get(i);

			if (abs(va - vb) > (atol + rtol * abs(vb))) {
				return false;
			}
		}

		return true;
	}

	template <typename T>
	bool allclose(Matrix<T>* a, Matrix<T>* b, double rtol) {
		return allclose(a, b, rtol, 1e-08);
	}

	template <typename T>
	bool allclose(Matrix<T>* a, Matrix<T>* b) {
		return allclose(a, b, 1e-05, 1e-08);
	}

	template <typename T>
	bool allclose(Matrix<T>* a, Vector<T>* b, double rtol, double atol) {
		// assert a width == b size

		for (int i = 0; i < a->getHeight(); i++) {
			for (int j = 0; j < a->getWidth(); j++) {
				T va = a->get(i, j);
				T vb = b->get(j);

				if (abs(va - vb) > (atol + rtol * abs(vb))) {
					return false;
				}
			}
		}

		return true;
	}

	template <typename T>
	bool allclose(Matrix<T>* a, Vector<T>* b, double rtol) {
		return allclose(a, b, rtol, 1e-08);
	}

	template <typename T>
	bool allclose(Matrix<T>* a, Vector<T>* b) {
		return allclose(a, b, 1e-05, 1e-08);
	}

	template <typename T>
	bool allclose(Vector<T>* vec, T val, double rtol, double atol) {
		for (int i = 0; i < vec->getSize(); i++) {
			if (abs(vec->get(i) - val) >(atol + rtol * abs(val))) {
				return false;
			}
		}

		return true;
	}

	template <typename T>
	bool allclose(Vector<T>* vec, T val, double rtol) {
		return allclose(vec, val, rtol, 1e-08);
	}

	template <typename T>
	bool allclose(Vector<T>* vec, T val) {
		return allclose(vec, val, 1e-05, 1e-08);
	}

	template <typename T>
	bool allclose(Matrix<T>* mat, T val, double rtol, double atol) {
		for (int i = 0; i < mat->getSize(); i++) {
			if (abs(mat->get(i) - val) > (atol + rtol * abs(val))) {
				return false;
			}
		}

		return true;
	}

	template <typename T>
	bool allclose(Matrix<T>* mat, T val, double rtol) {
		return allclose(mat, val, rtol, 1e-08);
	}

	template <typename T>
	bool allclose(Matrix<T>* mat, T val) {
		return allclose(mat, val, 1e-05, 1e-08);
	}

	template<typename T>
	bool equals(Vector<T>* a, Vector<T>* b) {
		if (a->getSize() != b->getSize()) {
			return false;
		}
		return allclose(a, b, 0, 0);
	}

	template<typename T>
	bool equals(Matrix<T>* a, Matrix<T>* b) {
		if (a->getWidth() != b->getWidth() || a->getHeight() != b->getHeight()) {
			return false;
		}
		return allclose(a, b, 0, 0);
	}
}