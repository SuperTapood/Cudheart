#pragma once

#include "../Util.cuh"
#include "Matrix.cuh"
#include "Vector.cuh"
#include "VectorOps.cuh"
#include "MatrixOps.cuh"
#include "../Exceptions/Exceptions.cuh"

using Cudheart::NDArrays::Matrix;
using Cudheart::NDArrays::Vector;
using Cudheart::NDArrays::NDArray;
using Cudheart::VectorOps::zeros;
using Cudheart::VectorOps::empty;
using namespace Cudheart::Exceptions;
using namespace Cudheart::MatrixOps;

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
		int newSize = a->getSize() + b->getSize();
		Vector<T>* vec = empty<T>(newSize);

		for (int i = 0; i < a->getSize(); i++) {
			vec->set(i, a->get(i));
		}

		for (int i = 0; i < b->getSize(); i++) {
			vec->set(i + a->getSize(), b->get(i));
		}

		return vec;
	}

	template <typename T>
	Matrix<T>* concatenate(Matrix<T>* a, Matrix<T>* b) {
		a->assertMatchShape(b->getShape());
		Matrix<T>* mat = MatrixOps::empty<T>(a->getHeight() * 2, a->getWidth());

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
		if (vec->getSize() % sizes != 0) {
			BadValueException("vector split does not result in an equal division");
			return nullptr;
		}

		int vecs = vec->getSize() / sizes;
		Vector<T>** out = (Vector<T>**)malloc(sizeof(Vector<T>*) * sizes);

		int index = 0;
		int jdex = 0;
		out[0] = new Vector<T>(vecs);
		for (int i = 0; i < vec->getSize(); i++, index++) {
			if (index == vecs) {
				index = 0;
				jdex++;
				out[jdex] = new Vector<T>(vecs);
			}
			((Vector<T>*)(out[jdex]))->set(index, vec->get(i));
		}

		return out;
	}

	template <typename T>
	Vector<T>** split(Matrix<T>* mat, int sizes) {
		// add axis parameter
		// currently assumes axis = 0

		if (mat->getHeight() % sizes != 0) {
			BadValueException("matrix split does not result in an equal division");
			return nullptr;
		}

		return split(mat->flatten(), sizes);
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
	Matrix<T>* tile(Matrix<T>* a, int hReps, int wReps) {
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

	template <typename T>
	Vector<T>* remove(Vector<T>* arr, int index, int axis) {
		Vector<T>* vec = new Vector<T>(arr->getSize() - 1);

		int idx = 0;
		for (int i = 0; i < arr->getSize(); i++) {
			if (i == index) {
				continue;
			}
			vec->set(idx++, arr->get(i));
		}

		return vec;
	}

	template <typename T>
	Matrix<T>* remove(Matrix<T>* arr, int index, int axis) {
		if (axis == 0) {
			Matrix<T>* mat = new Matrix<T>(arr->getHeight() - 1, arr->getWidth());

			int count = 0;
			for (int i = 0; i < arr->getHeight(); i++) {
				if (i == index) {
					continue;
				}

				for (int j = 0; j < arr->getWidth(); j++) {
					mat->set(count++, j, arr->get(i, j));
				}
			}

			return mat;
		}
		else if (axis == 1) {
			Matrix<T>* mat = new Matrix<T>(arr->getHeight(), arr->getWidth() - 1);

			for (int i = 0; i < arr->getHeight(); i++) {
				int count = 0;
				for (int j = 0; j < arr->getWidth(); j++) {
					if (j == index) {
						continue;
					}
					mat->set(i, count, arr->get(i, j));
				}
			}

			return mat;
		}
		return nullptr;
	}

	template <typename T>
	Vector<T>* trimZeros(NDArray<T>* filt, string trim = "fb") {
		int start = 0;
		int end = filt->getSize();
		if (trim == "fb") {
			for (; start < filt->getSize(); start++) {
				if (filt->get(start) != 0) {
					break;
				}
			}

			for (; end > start; end--) {
				if (filt->get(end) != 0) {
					break;
				}
			}
		}
		else if (trim == "f") {
			for (; start < filt->getSize(); start++) {
				if (filt->get(start) != 0) {
					break;
				}
			}
		}
		else if (trim == "b") {
			for (; end > start; end--) {
				if (filt->get(end) != 0) {
					break;
				}
			}
		}

		Vector<T>* vec = new Vector<T>(end - start);

		int index = 0;
		for (; start < end; start++) {
			vec->set(index++, filt->get(start));
		}

		return vec;
	}

	template <typename T>
	Vector<T>** unique(NDArray<T>* ar, bool returnIndex = false, bool returnInverse = false, bool returnCounts = false) {
		Vector<T>** vectors = new Vector<T>*[4];

		int countU = 1;
		for (int i = 1; i < ar->getSize(); i++) {
			bool unique = true;
			for (int j = 0; j < i; i++) {
				if (ar->get(j) == ar->get(i)) {
					unique = false;
					break;
				}
			}
			if (unique) {
				countU++;
			}
		}

		Vector<T>* uniqueVec = new Vector<T>(countU);
		Vector<T>* indexVec = returnIndex ? new Vector<T>(countU) : nullptr;
		Vector<T>* inverseVec = returnInverse ? new Vector<T>(ar->getSize()) : nullptr;
		Vector<T>* countVec = returnCounts ? zeros<T>(countU) : nullptr;

		// this is some mega big brain stuff
		// this has not been tested but if this works first try i will lose it

		int index = 1;
		uniqueVec->set(0, ar->get(0));
		if (returnIndex)
			indexVec->set(0, 0);
		if (returnInverse)
			inverseVec->set(0, 0);

		for (int i = 1; i < ar->getSize(); i++) {
			bool unique = true;
			int j = 0;
			for (; j < i; i++) {
				if (ar->get(j) == ar->get(i)) {
					unique = false;
					break;
				}
			}
			if (unique) {
				uniqueVec->set(index++, ar->get(i));
				if (returnIndex)
					indexVec->set(index++, i);
			}

			if (returnInverse)
				inverseVec->set(i, j);
			if (returnCounts)
				countVec->set(index, countVec->get(index) + 1);
		}

		vectors[0] = uniqueVec;
		vectors[1] = indexVec;
		vectors[2] = inverseVec;
		vectors[3] = countVec;

		return vectors;
	}
}