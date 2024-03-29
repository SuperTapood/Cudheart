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
using namespace Cudheart::Exceptions;
using namespace Cudheart::MatrixOps;

namespace Cudheart {
	namespace ArrayOps {
		template <typename T>
		Vector<T>* append(Vector<T>* a, T b) {
			Vector<T>* vec = new Vector<T>(a->size() + 1);

			for (int i = 0; i < a->size(); i++) {
				vec->set(i, a->get(i));
			}

			vec->set(a->size(), b);

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

			for (int i = 0; i < b->size(); i++) {
				mat->set(a->getHeight(), i, b->get(i));
			}

			return mat;
		}

		template <typename T>
		Vector<T>* concatenate(Vector<T>* a, Vector<T>* b) {
			int newSize = a->size() + b->size();
			Vector<T>* vec = new Vector<T>(newSize);

			for (int i = 0; i < a->size(); i++) {
				vec->set(i, a->get(i));
			}

			for (int i = 0; i < b->size(); i++) {
				vec->set(i + a->size(), b->get(i));
			}

			return vec;
		}

		template <typename T>
		Matrix<T>* concatenate(Matrix<T>* a, Matrix<T>* b) {
			a->assertMatchShape(b->getShape());
			Matrix<T>* mat = new Matrix<T>(a->getHeight() * 2, a->getWidth());

			for (int i = 0; i < a->size(); i++) {
				mat->set(i, a->get(i));
			}

			for (int i = 0; i < b->size(); i++) {
				mat->set(i + a->size(), b->get(i));
			}

			return mat;
		}

		template <typename T>
		Vector<T>** split(Vector<T>* vec, int sizes) {
			if (vec->size() % sizes != 0) {
				BadValueException("vector split does not result in an equal division");
				return nullptr;
			}

			int vecs = vec->size() / sizes;
			Vector<T>** out = new Vector<T>*[sizes];

			int index = 0;
			int jdex = 0;
			out[0] = new Vector<T>(vecs);
			for (int i = 0; i < vec->size(); i++, index++) {
				if (index == vecs) {
					index = 0;
					jdex++;
					out[jdex] = new Vector<T>(vecs);
				}
				((Vector<T>*)out[jdex])->set(index, vec->get(i));
			}

			return out;
		}

		template <typename T>
		Vector<T>** split(Matrix<T>* mat, int sizes) {
			// add axis parameter
			// currently assumes axis = 0

			/*if (mat->getHeight() % sizes != 0) {
				BadValueException("matrix split does not result in an equal division");
				return nullptr;
			}*/

			return split(mat->flatten(), sizes);
		}

		template <typename T>
		Vector<T>* tile(Vector<T>* a, int reps) {
			Vector<T>* out = new Vector<T>(a->size() * reps);

			for (int i = 0; i < a->size() * reps; i++) {
				out->set(i, a->get(i % a->size()));
			}

			return out;
		}

		template <typename T>
		Matrix<T>* tile(Vector<T>* a, int hReps, int wReps) {
			Matrix<T>* out = new Matrix<T>(hReps, a->size() * wReps);

			for (int i = 0; i < hReps; i++) {
				for (int j = 0; j < a->size() * wReps; j++) {
					out->set(i, j, a->get(j % a->size()));
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
		Vector<T>* remove(Vector<T>* arr, int index) {
			Vector<T>* vec = new Vector<T>(arr->size() - 1);

			int idx = 0;
			for (int i = 0; i < arr->size(); i++) {
				if (i == index) {
					continue;
				}
				vec->set(idx++, arr->get(i));
			}

			return vec;
		}

		template <typename T>
		NDArray<T>* remove(Matrix<T>* arr, int index, int axis = -1) {
			if (axis == 0) {
				Matrix<T>* mat = new Matrix<T>(arr->getHeight() - 1, arr->getWidth());

				int count = 0;

				for (int i = 0; i < mat->getHeight(); i++) {
					for (int j = 0; j < mat->getWidth(); j++) {
						if (i != index) {
							mat->set(i, j, arr->get(count++));
						}
					}
				}

				return mat;
			}
			else if (axis == 1) {
				Matrix<T>* mat = new Matrix<T>(arr->getHeight(), arr->getWidth() - 1);

				int idx = 0;
				for (int i = 0; i < arr->getHeight(); i++) {
					for (int j = 0; j < arr->getWidth(); j++) {
						if (j != index) {
							mat->set(idx++, arr->get(i, j));
						}
					}
				}

				return mat;
			}

			else if (axis == -1) {
				return remove((Vector<T>*)(arr->flatten()), index);
			}
			return nullptr;
		}

		template <typename T>
		Vector<T>* trimZeros(NDArray<T>* filt, string trim = "fb") {
			int start = 0;
			int end = filt->size();

			if (trim.find("f") != std::string::npos) {
				for (int i = 0; i < filt->size(); i++) {
					if (filt->get(i) != 0) {
						start = i;
						break;
					}
				}
			}

			if (trim.find("b") != std::string::npos) {
				for (int i = filt->size() - 1; i > start; i--) {
					if (filt->get(i) != 0) {
						end = i + 1;
						break;
					}
				}
			}

			int size = end - start;

			Vector<T>* out = new Vector<T>(size);

			for (int i = 0; i < size; i++) {
				out->set(i, filt->get(i + start));
			}

			return out;
		}

		namespace {
			// :,(
			template <typename T>
			int partition(NDArray<T>* arr, int low, int high) {
				T pivot = arr->get(high);
				int i = (low - 1);

				for (int j = low; j <= high - 1; j++)
				{
					// If current element is smaller than the pivot
					if (arr->get(j) < pivot)
					{
						i++; // increment index of smaller element
						T temp = arr->get(i);
						arr->set(i, arr->get(j));
						arr->set(j, temp);
					}
				}
				T temp = arr->get(i + 1);
				arr->set(i + 1, arr->get(high));
				arr->set(high, temp);
				return (i + 1);
			}

			template <typename T>
			void quicksort(NDArray<T>* arr, int low, int high) {
				if (low < high) {
					int pi = partition(arr, low, high);

					quicksort(arr, low, pi - 1);
					quicksort(arr, pi + 1, high);
				}
			}
		}

		template <typename T>
		Vector<T>** unique(NDArray<T>* ar, bool returnIndex, bool returnInverse, bool returnCounts) {
			Vector<T>** vectors = new Vector<T>*[4];

			int uniques = 0;

			for (int i = 0; i < ar->size(); i++) {
				bool isUnique = true;
				for (int j = 0; j < i; j++) {
					if (ar->get(i) == ar->get(j)) {
						isUnique = false;
					}
				}

				if (isUnique) {
					uniques++;
				}
			}

			auto uniqueArr = new Vector<T>(uniques);
			auto indexArr = new Vector<T>(uniques);
			auto inverseArr = new Vector<T>(ar->size());
			auto countsArr = zeros<T>(uniques);

			int index = 0;

			for (int i = 0; i < ar->size(); i++) {
				bool unique = true;
				for (int j = 0; j < index; j++) {
					if (uniqueArr->get(j) == ar->get(i)) {
						unique = false;
						break;
					}
				}

				if (unique) {
					uniqueArr->set(index++, ar->get(i));

					if (index == uniques) {
						break;
					}
				}
			}

			quicksort(uniqueArr, 0, uniques - 1);

			if (returnIndex) {
				for (int i = 0; i < uniqueArr->size(); i++) {
					for (int j = 0; j < ar->size(); j++) {
						if (ar->get(j) == uniqueArr->get(i)) {
							indexArr->set(i, j);
							break;
						}
					}
				}
			}

			if (returnInverse) {
				for (int i = 0; i < ar->size(); i++) {
					for (int j = 0; j < uniqueArr->size(); j++) {
						if (ar->get(i) == uniqueArr->get(j)) {
							inverseArr->set(i, j);
						}
					}
				}
			}

			if (returnCounts) {
				for (int i = 0; i < ar->size(); i++) {
					for (int j = 0; j < uniqueArr->size(); j++) {
						if (ar->get(i) == uniqueArr->get(j)) {
							countsArr->set(j, countsArr->get(j) + 1);
						}
					}
				}
			}

			vectors[0] = uniqueArr;
			vectors[1] = returnIndex ? indexArr : nullptr;
			vectors[2] = returnInverse ? inverseArr : nullptr;
			vectors[3] = returnCounts ? countsArr : nullptr;

			return vectors;
		}

		template <typename T>
		Vector<T>** unique(NDArray<T>* ar, bool returnIndex, bool returnInverse) {
			return unique<T>(ar, returnIndex, returnInverse, false);
		}

		template <typename T>
		Vector<T>** unique(NDArray<T>* ar, bool returnIndex) {
			return unique<T>(ar, returnIndex, false, false);
		}

		template <typename T>
		Vector<T>** unique(NDArray<T>* ar) {
			return unique<T>(ar, false, false, false);
		}
	}
}