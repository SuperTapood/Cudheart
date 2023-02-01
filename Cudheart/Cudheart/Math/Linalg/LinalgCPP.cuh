#pragma once

#include "../../Arrays/Arrays.cuh"

using Cudheart::NDArrays::Vector;
using Cudheart::NDArrays::Matrix;
using namespace Cudheart::Exceptions;
using Cudheart::VectorOps::emptyLike;
using Cudheart::MatrixOps::emptyLike;
using Cudheart::ArrayOps::append;
using Cudheart::MatrixOps::fromVectorArray;

namespace Cudheart::CPP::Math::Linalg {
	template <typename T>
	T dot(Vector<T>* a, Vector<T>* b) {
		a->assertMatchShape(b->getShape());

		T result = (T)0;

		for (int i = 0; i < a->getSize(); i++) {
			result += a->get(i) * b->get(i);
		}

		return result;
	}

	template <typename T>
	Vector<T>* dot(Matrix<T>* a, Vector<T>* b) {
		a->assertMatchShape(b->getShape(), 0);

		Vector<T>* out = new Vector<T>(a->getHeight());

		for (int i = 0; i < a->getHeight(); i++) {
			T sum = (T)0;
			for (int j = 0; j < a->getWidth(); j++) {
				sum += (a->get(i, j) * b->get(j));
			}
			out->set(i, sum);
		}

		return out;
	}

	template <typename T>
	Vector<T>* dot(Vector<T>* a, Matrix<T>* b) {
		b->assertMatchShape(a->getShape(), 1);

		auto out = new Vector<T>(b->getWidth());

		for (int i = 0; i < b->getWidth(); i++) {
			T sum = (T)0;
			for (int j = 0; j < b->getHeight(); j++) {
				sum += (a->get(j) * b->get(j, i));
			}
			out->set(i, sum);
		}

		return out;
	}

	template <typename T>
	Matrix<T>* dot(Matrix<T>* a, Matrix<T>* b) {
		if (a->getWidth() != b->getHeight()) {
			if (a->getHeight() != b->getWidth()) {
				Cudheart::Exceptions::ShapeMismatchException(a->getShape()->toString(),
					b->getShape()->toString()).raise();
			}
			else {
				return dot(b, a);
			}
		}
		
		Matrix<T>* out = new Matrix<T>(a->getHeight(), b->getWidth());

		int idx;

		for (int i = 0; i < out->getHeight(); i++) {
			for (int j = 0; j < out->getWidth(); j++) {
				T sum = (T)0;
				for (int k = 0; k < a->getWidth(); k++) {
					sum += a->get(i, k) * b->get(k, j);
				}
				out->set(i, j, sum);
			}
		}

		return out;
	}

	template <typename T>
	T inner(Vector<T>* a, Vector<T>* b) {
		return dot(a, b);
	}

	template <typename T>
	Vector<T>* inner(Vector<T>* vec, Matrix<T>* mat) {
		return dot(mat, vec);
	}

	template <typename T>
	Vector<T>* inner(Matrix<T>* mat, Vector<T>* vec) {
		return dot(mat, vec);
	}

	template <typename T>
	Matrix<T>* inner(Matrix<T>* a, Matrix<T>* b) {
		return dot(a, (Matrix<T>*)b->transpose());
	}


	template <typename T>
	Matrix<T>* outer(Vector<T>* a, Vector<T>* b) {
		Matrix<T>* out = new Matrix<T>(a->getSize(), b->getSize());

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

		Matrix<T>* out = new Matrix<T>(a->getSize(), b->getSize());

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
		Matrix<T>* out = new Matrix<T>(mat->getSize(), vec->getSize());

		for (int i = 0; i < mat->getSize(); i++) {
			for (int j = 0; j < vec->getSize(); j++) {
				out->set(i, j, mat->get(i) * vec->get(j));
			}
		}

		return out;
	}

	template <typename T>
	Matrix<T>* outer(Vector<T>* vec, Matrix<T>* mat) {
		Matrix<T>* out = new Matrix<T>(vec->getSize(), mat->getSize());

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
			BaseException("Exception: Matrix has to be square").raise();
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
			Matrix<T>* sub = new Matrix<T>(mat->getHeight() - 1, mat->getWidth() - 1);
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
		T value = (T)0;

		for (int i = 0; i < mat->getHeight() && i + offset < mat->getWidth(); i++) {
			value += mat->get(i, i + offset);
		}

		return value;
	}

	template <typename T>
	T trace(Matrix<T>* mat) {
		return trace(mat, 0);
	}

	template <typename T>
	Vector<T>* solve(Matrix<T>* a, Vector<T>* b) {
		a->assertMatchShape(b->getShape(), 1);
		// relentlessly ripped from https://martin-thoma.com/solving-linear-equations-with-gaussian-elimination/

		Matrix<T>* A = new Matrix<T>(a->getHeight(), a->getWidth() + 1);

		for (int i = 0; i < a->getHeight(); i++) {
			for (int j = 0; j < a->getWidth(); j++) {
				A->set(i, j, a->get(i, j));
			}
		}

		for (int i = 0; i < b->getSize(); i++) {
			A->set(i, -1, b->get(i));
		}

		int n = a->getHeight();

		for (int i = 0; i < n; i++) {
			// Search for maximum in this column
			double maxEl = abs(A->get(i, i));
			int maxRow = i;
			for (int k = i + 1; k < n; k++) {
				if (abs(A->get(k, i)) > maxEl) {
					maxEl = abs(A->get(k, i));
					maxRow = k;
				}
			}

			// Swap maximum row with current row (column by column)
			for (int k = i; k < n + 1; k++) {
				double tmp = A->get(maxRow, k);
				A->set(maxRow, k, A->get(i, k));
				A->set(i, k, tmp);
			}

			// Make all rows below this one 0 in current column
			for (int k = i + 1; k < n; k++) {
				double c = -A->get(k, i) / A->get(i, i);
				for (int j = i; j < n + 1; j++) {
					if (i == j) {
						A->set(k, j, 0);
					}
					else {
						A->set(k, j, A->get(k, j) + (c * A->get(i, j)));
					}
				}
			}
		}

		// Solve equation Ax=b for an upper triangular matrix A
		// vector<double> x(n);
		Vector<T>* x = new Vector<T>(n);
		for (int i = n - 1; i >= 0; i--) {
			x->set(i, A->get(i, n) / A->get(i, i));
			for (int k = i - 1; k >= 0; k--) {
				A->set(k, n, A->get(k, n) - (A->get(k, i) * x->get(i)));
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