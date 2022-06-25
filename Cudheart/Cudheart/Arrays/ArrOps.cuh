#pragma once

#include "../Util.cuh"
#include "Matrix.cuh"
#include "Vector.cuh"
#include "VectorOps.cuh"
#include "../Exceptions/Exceptions.cuh"

using Cudheart::NDArrays::Matrix;
using Cudheart::NDArrays::Vector;
using namespace Cudheart::Exceptions;

namespace Cudheart::ArrayOps {
	template <typename T>
	Vector<T>* append(Vector<T>* a, T b) {
		Vector<T>* vec = new Vector<T>(a->getSize() + 1);

		for (int i = 0; i < a->getSize(); i++) {
			vec->set(i, a->get(i));
		}

		vec->set(a->getSize(), b);

		return vec;
	}

	template <typename T>
	Matrix<T>* append(Matrix<T>* a, Vector<T>* b) {
		/// assert axis match
		a->assertMatchShape(b->getShape());
		Matrix<T>* mat = new Matrix<T>(a->getHeight() + 1, a->getWidth());

		for (int i = 0; i < a->getHeight(); i++) {
			for (int j = 0; j < a->getWidth(); j++) {
				mat->set(i, j, a->get(i, j));
			}
		}

		for (int i = 0; i < b->getSize(); i++) {
			mat->set(a->getHeight(), i, b->get(i));
		}

		return mat;
	}

	template <typename T>
	Vector<T>* concatenate(Vector<T>* a, Vector<T>* b) {
		Vector<T>* vec = new Vector(a->getSize() + b->getSize());

		for (int i = 0; i < a->getSize(); i++) {
			vec->set(i, a->get(i));
		}

		for (int i = 0; i < b->getSize(); i++) {
			vec->set(i, b->get(i));
		}

		return vec;
	}

	template <typename T>
	Matrix<T>* concatenate(Matrix<T>* a, Matrix<T>* b) {
		a->assertMatchShape(b);
		Matrix<T>* mat = new Matrix(a->getHeight() * 2, a->getWidth());

		for (int i = 0; i < a->getSize(); i++) {
			mat->set(i, a->get(i));
		}

		for (int i = 0; i < b->getSize(); i++) {
			mat->set(i + a->getSize(), b->get(i));
		}

		return mat;
	}

	template <typename T>
	Vector<T>** split(Vector<T>* vec, int sizes) {
		int len = ceil(vec->getSize() / sizes);
		Vector<T>** out = (Vector<T>**)malloc(sizeof(Vector<T>*) * len);

		int count = vec->getSize();
		for (int i = 0; i < len; i++) {
			if (count > sizes) {
				out[i] = new Vector<T>(sizes);
				count -= sizes;
			}
			else {
				out[i] = new Vector<T>(count);
				break;
			}
		}

		int vectorIndex = 0;
		int elems = 0;
		Vector<T>* v = out[vectorIndex];
		for (int i = 0; i < vec->getSize(); i++) {
			v->set(elems, vec->get(i));
			elems++;
			if (elems == sizes) {
				elems = 0;
				Vector<T>* v = out[vectorIndex++];
			}
		}

		return out;
	}

	template <typename T>
	Matrix<T>** split(Matrix<T>* mat, int sizes) {
		// add axis parameter
		// currently assumes axis = 0

		int len = ceil(mat->getWidth() / sizes);
		Matrix<T>** out = (Matrix<T>**)malloc(sizeof(Matrix<T>*) * len);
		
		int count = mat->getWidth();
		for (int i = 0; i < len; i++) {
			if (count > sizes) {
				out[i] = new Matrix<T>(mat->getHeight(), sizes);
				count -= sizes;
			}
			else {
				out[i] = new Matrix<T>(mat->getHeight(), count);
				break;
			}
		}

		int matrixIndex = 0;
		int elems = 0;
		Matrix<T>* m = out[matrixIndex];
		for (int i = 0; i < mat->getWidth(); i++) { 
			for (int j = 0; j < mat->getHeight(); j++) {
				m->set(elems, mat->get(j, i));
			}
			elems++;
			if (elems == sizes) {
				elems = 0;
				Matrix<T>* m = out[matrixIndex++];
			}
		}

		return out;
	}

	template <typename T>
	Vector<T>* tile(Vector<T>* a, int reps) {
		Vector<T>* out = new Vector<T>(a->getSize() * reps);

		for (int i = 0; i < a->getSize() * reps; i++) {
			out->set(i, a->get(i % a->getSize()));
		}

		return out;
	}

	template <typename T>
	Matrix<T>* tile(Vector<T>* a, int wReps, int hReps) {
		Matrix<T>* out = new Matrix<T>(hReps, a->getSize() * wReps);

		for (int i = 0; i < hReps; i++) {
			for (int j = 0; j < a->getSize() * wReps; j++) {
				out->set(i, j, a->get(j % a->getSize()));
			}
		}

		return out;
	}

	template <typename T>
	Matrix<T>* tile(Matrix<T>* a, int reps) {
		// add axis
		// assumes axis = 0
		Matrix<T>* out = new Matrix<T>(a->getHeight(), a->getWidth() * reps);

		for (int i = 0; i < out->getHeight(); i++) {
			for (int j = 0; j < out->getWidth(); j++) {
				out->set(i, j, a->get(i, j % a->getWidth()));
			}
		}

		return out;
	}

	template <typename T>
	Matrix<T>* tile(Matrix<T>* a, int hReps, int rReps) {
		// add axis
		// assumes axis = 0
		Matrix<T>* out = new Matrix<T>(a->getHeight() * hReps, a->getWidth() * wReps);

		for (int i = 0; i < out->getHeight(); i++) {
			for (int j = 0; j < out->getWidth(); j++) {
				out->set(i, j, a->get(i % a->getHeight(), j % a->getWidth()));
			}
		}

		return out;
	}
}