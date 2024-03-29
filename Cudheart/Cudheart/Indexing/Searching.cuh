#pragma once

#include "../Arrays/Shape.cuh"
#include "../Arrays/Arrays.cuh"

using namespace Cudheart::NDArrays;

namespace Cudheart::Searching {
	template <typename T>
	int argmax(NDArray<T>* a) {
		int index = 0;
		T value = a->get(0);

		for (int i = 1; i < a->size(); i++) {
			if (a->get(i) > value) {
				value = a->get(i);
				index = i;
			}
		}

		return index;
	}

	template <typename T>
	int argmin(NDArray<T>* a) {
		int index = 0;
		T value = a->get(0);

		for (int i = 1; i < a->size(); i++) {
			if (a->get(i) < value) {
				value = a->get(i);
				index = i;
			}
		}

		return index;
	}

	template <typename T>
	Vector<T>* nonzero(Vector<T>* vec) {
		int count = 0;

		for (int i = 0; i < vec->size(); i++) {
			if (vec->get(i) != 0) {
				count++;
			}
		}

		Vector<T>* result = new Vector<T>(count);

		int index = 0;
		for (int i = 0; i < vec->size(); i++) {
			if (vec->get(i) != 0) {
				result->set(index, i);
				index++;
			}
		}

		return result;
	}

	template <typename T>
	Matrix<T>* nonzero(Matrix<T>* mat) {
		int count = 0;

		for (int i = 0; i < mat->size(); i++) {
			if (mat->get(i) != 0) {
				count++;
			}
		}

		Matrix<T>* result = new Matrix<T>(2, count);

		int index = 0;
		for (int i = 0; i < mat->getHeight(); i++) {
			for (int j = 0; j < mat->getWidth(); j++) {
				if (mat->get(i, j) != 0) {
					result->set(0, index, i);
					result->set(1, index, j);
					index++;
				}
			}
		}

		return result;
	}

	template <typename T>
	Matrix<T>* argwhere(Vector<T>* a) {
		Vector<T>* vec = nonzero(a);
		return (Matrix<T>*)(vec->reshape(new Shape(vec->size(), 1)));
	}

	template <typename T>
	Matrix<T>* argwhere(Matrix<T>* a) {
		Matrix<T>* n = nonzero(a);
		return (Matrix<T>*)n->transpose(true);
	}

	template <typename T>
	Vector<T>* flatnonzero(NDArray<T>* a) {
		return nonzero((Vector<T>*)a->flatten());
	}

	template <typename T>
	inline NDArray<T>* where(NDArray<bool>* condition, NDArray<T>* x, NDArray<T>* y) {
		x->assertMatchShape(condition->getShape());
		x->assertMatchShape(y->getShape());

		NDArray<T>* out = x->emptyLike();

		for (int i = 0; i < x->size(); i++) {
			if (condition->get(i)) {
				out->set(i, x->get(i));
			}
			else {
				out->set(i, y->get(i));
			}
		}

		return out;
	}

	template <typename T>
	inline int searchsorted(Vector<T>* a, T v, string side, Vector<int>* sorter) {
		a->assertMatchShape(sorter->getShape());
		if (side == "left") {
			for (int i = 1; i < a->size(); i++) {
				if (a->get(sorter->get(i - 1)) < v && v <= a->get(sorter->get(i))) {
					return i;
				}
			}
		}
		else if (side == "right") {
			for (int i = 1; i < a->size(); i++) {
				if (a->get(sorter->get(i - 1)) <= v && v < a->get(sorter->get(i))) {
					return i;
				}
			}
		}

		if (v > a->get(0)) {
			return a->size();
		}

		return 0;
	}

	template <typename T>
	int searchsorted(Vector<T>* a, T v, string side) {
		if (side == "left") {
			for (int i = 1; i < a->size(); i++) {
				if (a->get(i - 1) < v && v <= a->get(i)) {
					return i;
				}
			}
		}
		else if (side == "right") {
			for (int i = a->size() - 1; i >= 0; i--) {
				if (a->get(i - 1) <= v && v < a->get(i)) {
					return i;
				}
			}
		}

		if (v > a->get(0)) {
			return a->size();
		}

		return 0;
	}

	template <typename T>
	inline int searchsorted(Vector<T>* a, T v, Vector<int>* sorter) {
		return searchsorted(a, v, "left", sorter);
	}

	template <typename T>
	int searchsorted(Vector<T>* a, T v) {
		return searchsorted(a, v, "left");
	}

	template <typename T>
	inline Vector<int>* searchsorted(Vector<T>* a, Vector<T>* v, string side, Vector<int>* sorter) {
		Vector<int>* out = new Vector<int>(v->size());

		for (int i = 0; i < out->size(); i++) {
			out->set(i, searchsorted(a, v->get(i), side, sorter));
		}

		return out;
	}

	template <typename T>
	Vector<int>* searchsorted(Vector<T>* a, Vector<T>* v, string side) {
		Vector<int>* out = new Vector<int>(v->size());

		for (int i = 0; i < out->size(); i++) {
			out->set(i, searchsorted(a, v->get(i), side));
		}

		return out;
	}

	template <typename T>
	inline Vector<int>* searchsorted(Vector<T>* a, Vector<T>* v, Vector<int>* sorter) {
		return searchsorted(a, v, "left", sorter);
	}

	template <typename T>
	Vector<int>* searchsorted(Vector<T>* a, Vector<T>* v) {
		return searchsorted(a, v, "left");
	}

	template <typename T>
	inline Vector<T>* extract(Vector<bool>* condition, Vector<T>* arr) {
		condition->assertMatchShape(arr->getShape());
		int size = 0;

		for (int i = 0; i < condition->size(); i++) {
			if (condition->get(i)) {
				size++;
			}
		}

		Vector<T>* vec = new Vector<T>(size);

		int index = 0;
		for (int i = 0; i < arr->size(); i++) {
			if (condition->get(i)) {
				vec->set(index, arr->get(i));
				index++;
			}
		}

		return vec;
	}

	template <typename T>
	inline Vector<T>* extract(Matrix<bool>* condition, Matrix<T>* arr) {
		return extract(condition->flatten(), arr->flatten());
	}

	template <typename T>
	int count_nonzero(NDArray<T>* a) {
		int count = 0;

		for (int i = 0; i < a->size(); i++) {
			if (a->get(i) != 0) {
				count++;
			}
		}

		return count;
	}
}