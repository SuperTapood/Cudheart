#pragma once

#include "../../Arrays/Arrays.cuh"

using Cudheart::NDArrays::Vector;
using Cudheart::NDArrays::Matrix;
using namespace Cudheart::Exceptions;
using Cudheart::VectorOps::empty;
using Cudheart::VectorOps::emptyLike;
using Cudheart::MatrixOps::empty;
using Cudheart::MatrixOps::emptyLike;
using Cudheart::MatrixOps::fromVectorArray;


namespace Cudheart::CPP::Math::Linalg {
	template <typename T>
	T dot(Vector<T>* a, Vector<T>* b) {
		a->assertMatchSize(b);

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
		if (a->getWidth() != b->getSize()) {
			ShapeMismatchException(a->getWidth(), b->getSize()).raise();
		}

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
		a->assertMatchSize(b);

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
		Vector<T>* va = a->flatten();
		Vector<T>* vb = b->flatten();

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
}