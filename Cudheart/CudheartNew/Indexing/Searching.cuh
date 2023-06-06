#pragma once

#include "../Arrays/NDArray.cuh"

#include "../Internal/Broadcasting.cuh"
#include "../Internal/Promotion.cuh"

namespace CudheartNew::Searching {
	template <typename T>
	int argmax(NDArray<T>* a) {
		int index = 0;
		T value = a->at(0);

		for (int i = 1; i < a->size(); i++) {
			if (a->at(i) > value) {
				value = a->at(i);
				index = i;
			}
		}

		return index;
	}

	template <typename T>
	int argmin(NDArray<T>* a) {
		int index = 0;
		T value = a->at(0);

		for (int i = 1; i < a->size(); i++) {
			if (a->at(i) < value) {
				value = a->at(i);
				index = i;
			}
		}

		return index;
	}

	template <typename T>
	NDArray<T>* nonzero(NDArray<T>* arr) {
		int count = 0;

		for (int i = 0; i < arr->size(); i++) {
			if (arr->at(i) != 0) {
				count++;
			}
		}

		NDArray<T>* result = new NDArray<T>({ count });

		int index = 0;
		for (int i = 0; i < arr->size(); i++) {
			if (arr->at(i) != 0) {
				result->at(index) = i;
				index++;
			}
		}

		return result;
	}

	template <typename T>
	NDArray<T>* argwhere(NDArray<T>* a) {
		NDArray<T>* arr = nonzero<T>(a);
		return arr->transpose();
	}

	template <typename T>
	NDArray<T>* flatnonzero(NDArray<T>* a) {
		return nonzero<T>(a->flatten());
	}

	template <typename A, typename B, typename T = promote(A, B)>
	inline NDArray<T>* where(NDArray<bool>* condition, NDArray<A>* x, NDArray<B>* y) {
		auto tup = broadcast(std::make_tuple(condition, x, y));

		condition = std::get<0>(tup);
		x = std::get<1>(tup);
		y = std::get<2>(tup);

		auto out = new NDArray<T>(x->shape());

		for (int i = 0; i < x->size(); i++) {
			if (condition->at(i)) {
				out->at(i) = x->at(i);
			}
			else {
				out->at(i) = y->at(i);
			}
		}

		return out;
	}

	/*template <typename T>
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
	}*/
}