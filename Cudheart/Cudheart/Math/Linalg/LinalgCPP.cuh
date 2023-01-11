#pragma once

#include "../../Arrays/Arrays.cuh"

using Cudheart::NDArrays::Vector;
using Cudheart::NDArrays::Matrix;
using namespace Cudheart::Exceptions;
using Cudheart::VectorOps::emptyLike;
using Cudheart::MatrixOps::empty;
using Cudheart::MatrixOps::emptyLike;
using Cudheart::MatrixOps::fromVectorArray;

namespace Cudheart::CPP::Math::Linalg {
	template <typename T>
	T dot(Vector<T>* a, Vector<T>* b) {
		a->assertMatchShape(b->getShape());

		T result = 0;

		for (int i = 0; i < a->getSize(); i++) {
			result += a->get(i) * b->get(i);
		}

		return result;
	}

	template <typename T>
	Vector<T>* dot(Vector<T>* a, Matrix<T>* b) {
		if (b->getHeight() != a->getSize()) {
			ShapeMismatchException(b->getHeight(), a->getSize()).raise();
		}

		Matrix<T>* t = emptyLike<T>(b);

		for (int i = 0; i < b->getHeight(); i++) {
			for (int j = 0; j < b->getWidth(); j++) {
				t->set(i, j, b->get(i, j) * a->get(i));
			}
		}

		Vector<T>* out = emptyLike<T>(a);

		for (int i = 0; i < b->getWidth(); i++) {
			int sum = 0;
			for (int j = 0; j < b->getHeight(); j++) {
				sum += t->get(j, i);
			}
			out->set(i, sum);
		}

		delete t;

		return out;
	}

	template <typename T>
	Vector<T>* dot(Matrix<T>* a, Vector<T>* b) {
		a->assertMatchShape(b->getShape(), 1);

		Matrix<T>* t = emptyLike<T>(a);

		for (int i = 0; i < a->getHeight(); i++) {
			for (int j = 0; j < a->getWidth(); j++) {
				t->set(i, j, a->get(i, j) * b->get(j));
			}
		}

		Vector<T>* out = empty<T>(a->getHeight());

		for (int i = 0; i < a->getHeight(); i++) {
			int sum = 0;
			for (int j = 0; j < a->getWidth(); j++) {
				sum += t->get(i, j);
			}
			out->set(i, sum);
		}

		return out;
	}

	template <typename T>
	Vector<T>* inner(Matrix<T>* mat, Vector<T>* vec) {
		return dot(mat, vec);
	}

	template <typename T>
	Vector<T>* inner(Vector<T>* vec, Matrix<T>* mat) {
		return dot(mat, vec);
	}

	template <typename T>
	Matrix<T>* inner(Matrix<T>* a, Matrix<T>* b) {
		a->assertMatchShape(b->getShape());

		Matrix<T>* out = emptyLike<T>(a);

		for (int i = 0; i < a->getHeight(); i++) {
			Vector<T>* v = dot<T>(b, a->getRow(i));
			for (int j = 0; j < v->getSize(); j++) {
				out->set(i, j, v->get(j));
			}
		}

		return out;
	}

	template <typename T>
	Matrix<T>* outer(Vector<T>* a, Vector<T>* b) {
		Matrix<T>* out = empty<T>(a->getSize(), b->getSize());

		for (int i = 0; i < a->getSize(); i++) {
			for (int j = 0; j < b->getSize(); j++) {
				out->set(i, j, a->get(i) * b->get(j));
			}
		}

		return out;
	}

	template <typename T>
	Matrix<T>* outer(Matrix<T>* a, Matrix<T>* b) {
		Vector<T>* va = (Vector<T>*)a->flatten();
		Vector<T>* vb = (Vector<T>*)b->flatten();

		Matrix<T>* out = empty<T>(a->getSize(), b->getSize());

		for (int i = 0; i < va->getSize(); i++) {
			for (int j = 0; j < vb->getSize(); j++) {
				out->set(i, j, va->get(i) * vb->get(j));
			}
		}

		delete va, vb;

		return out;
	}

	template <typename T>
	Matrix<T>* outer(Matrix<T>* mat, Vector<T>* vec) {
		Matrix<T>* out = empty<T>(mat->getSize(), vec->getSize());

		for (int i = 0; i < mat->getSize(); i++) {
			for (int j = 0; j < vec->getSize(); j++) {
				out->set(i, j, mat->get(i) * vec->get(j));
			}
		}

		return out;
	}

	template <typename T>
	Matrix<T>* outer(Vector<T>* vec, Matrix<T>* mat) {
		Matrix<T>* out = empty<T>(vec->getSize(), mat->getSize());

		for (int i = 0; i < vec->getSize(); i++) {
			for (int j = 0; j < mat->getSize(); j++) {
				out->set(i, j, vec->get(i) * mat->get(j));
			}
		}

		return out;
	}

	template <typename T>
	T det(Matrix<T>* mat) {
		// idfk how numpy implemented their determinant algorithm
		// this is a mirror of the algorithm implemented in a plethera of websites
		if (mat->getHeight() != mat->getWidth()) {
			NotImplementedException("det needs custom exception").raise();
		}

		if (mat->getHeight() == 1) {
			return mat->get(0, 0);
		}

		if (mat->getHeight() == 2) {
			return (mat->get(0, 0) * mat->get(1, 1)) - (mat->get(0, 1) * mat->get(1, 0));
		}

		T value = 0;
		int sign = 1;

		for (int i = 0; i < mat->getWidth(); i++) {
			//mat->print();
			//cout << "value: " << mat->get(0, i) << " of index " << i << endl;
			Matrix<T>* sub = empty<T>(mat->getHeight() - 1, mat->getWidth() - 1);
			int idx = 0;
			int jdx = 0;
			for (int k = 1; k < mat->getHeight(); k++) {
				for (int m = 0; m < mat->getWidth(); m++) {
					if (m != i) {
						//cout << "m: " << m << " k: " << k << " " << mat->get(k, m) << endl;
						//cout << "i: " << idx << " j: " << jdx << endl;
						sub->set(jdx, idx, mat->get(k, m));
						idx++;
						if (idx == mat->getHeight() - 1) {
							idx = 0;
							jdx++;
						}
					}
				}
			}
			value += sign * mat->get(0, i) * det(sub);
			sign *= -1;
		}

		return value;
	}

	template <typename T>
	T trace(Matrix<T>* mat, int offset) {
		T value = mat->get(offset, offset);

		for (int i = offset + 1; i < mat->getHeight(); i++) {
			value += mat->get(i, i);
		}

		return value;
	}

	template <typename T>
	T trace(Matrix<T>* mat) {
		return trace(mat, 0);
	}

	template <typename T>
	Vector<T>* solve(Matrix<T>* a, Vector<T>* b) {
		// a is the coefficients matrix
		// b is the dependent variable (the ordinate) vector
		a->assertMatchShape(b->getShape(), 1);
		// assert a->getShape()->getX() == a->getShape()->getY();

		// relentlessly ripped from https://martin-thoma.com/solving-linear-equations-with-gaussian-elimination/

		int n = b->getSize();

		Matrix<T>* A = a->augment(b);

		Vector<T>* x = new Vector<T>(n);

		for (int i = 1; i < n; i++) {
			// search for maximum in this column

			T maxEl = A->get(i, i);
			int maxRow = i;

			for (int k = i + 1; k <= n; k++) {
				if (A->get(k, i) > maxEl) {
					maxEl = A->get(k, i);
					maxRow = k;
				}
			}

			// swap maximum row with current row
			for (int k = i; k <= n; k++) {
				T tmp = A->get(maxRow, k);
				A->set(maxRow, k, A->get(i, k));
				A->set(i, k, tmp);
			}

			// make all rows below this one 0 in current column
			for (int k = i + 1; k <= n; k++) {
				T c = -(A->get(k, i) / A->get(i, i));

				for (int j = i; j <= n; j++) {
					if (i == j) {
						A->set(k, j, 0);
					}
					else {
						A->set(k, j, A->get(k, j) + c * A->get(i, j));
					}
				}
			}
		}

		// solve equation for an upper triangular matrix
		for (int i = n - 1; i >= 0; i--) {
			T v = A->get(i, n) / A->get(i, i);
			x->set(i, v);
			for (int k = i - 1; k >= 0; k--) {
				T value = A->get(k, n) - (A->get(k, i) * v);
				A->set(k, n, value);
			}
		}

		return x;
	}

	template <typename T>
	Matrix<T>* inv(Matrix<T>* mat) {
		// find x where
		// dot(mat, x) = eye(a.shape[0])
		// or rather
		// mat * x = eye(a.shape[0])
		// and solved for x as such:
		// (eye(a.shape[0])) / mat = x

		Matrix<T>* eye = Cudheart::MatrixOps::eye(mat->getHeight());

		for (int i = 0; i < mat->getSize(); i++) {
			eye->set(i, (eye->get(i) / mat->get(i)));
		}

		return eye;
	}

	template <typename T>
	Vector<T>* convolve(Vector<T>* a, Vector<T>* b) {
		Vector<T>* out = Cudheart::VectorOps::zeros(a->getSize() + b->getSize() - 1);

		for (int i = 0; i < a->getSize(); i++) {
			for (int j = 0; j < b->getSize(); j++) {
				T last = out->get(i + j);
				out->set(i + j, last + (a->get(i) * b->get(j)));
			}
		}

		return out;
	}

	template <typename T>
	NDArray<T>* clip(NDArray<T>* arr, T min, T max) {
		NDArray<T>* out = arr->copy();

		for (int i = 0; i < out->getSize(); i++) {
			if (out->get(i) < min) {
				out->set(i, min);
			}
			else if (out->get(i) > max) {
				out->set(i, max);
			}
		}

		return out;
	}

	template <typename T>
	NDArray<T>* clip(NDArray<T>* arr, T max) {
		NDArray<T>* out = arr->copy();

		for (int i = 0; i < out->getSize(); i++) {
			if (out->get(i) > max) {
				out->set(i, max);
			}
		}

		return out;
	}
}