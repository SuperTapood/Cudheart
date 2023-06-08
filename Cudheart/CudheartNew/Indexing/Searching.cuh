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
		auto casted = broadcast({ condition, x, y });

		condition = (NDArray<bool>*)casted.at(0);
		x = (NDArray<A>*)casted.at(1);
		y = (NDArray<B>*)casted.at(2);

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

	template <typename T>

	int searchsorted(NDArray<T>* a, T v, std::string& side, NDArray<int>* sorter) {
		auto casted = broadcast({ a, sorter });

		a = (NDArray<T>*)casted.at(0);
		sorter = (NDArray<int>*)casted.at(1);

		if (side == "left") {
			for (int i = 1; i < a->size(); i++) {
				if (a->at(sorter->at(i - 1)) < v && v <= a->at(sorter->at(i))) {
					return i;
				}
			}
		}
		else if (side == "right") {
			for (int i = 1; i < a->size(); i++) {
				if (a->at(sorter->at(i - 1)) <= v && v < a->at(sorter->at(i))) {
					return i;
				}
			}
		}

		if (v > a->at(0)) {
			return a->size();
		}

		return -1;
	}

	template <typename T>
	int searchsorted(NDArray<T>* a, T v, std::string& side) {
		if (side == "left") {
			for (int i = 1; i < a->size(); i++) {
				if (a->at(i - 1) < v && v <= a->at(i)) {
					return i;
				}
			}
		}
		else if (side == "right") {
			for (int i = a->size() - 1; i >= 0; i--) {
				if (a->at(i - 1) <= v && v < a->at(i)) {
					return i;
				}
			}
		}

		if (v > a->at(0)) {
			return a->size();
		}

		return -1;
	}

	template <typename T>
	inline int searchsorted(NDArray<T>* a, T v, NDArray<int>* sorter) {
		return searchsorted(a, v, "left", sorter);
	}

	template <typename T>
	int searchsorted(NDArray<T>* a, T v) {
		return searchsorted(a, v, "left");
	}

	template <typename T>
	inline NDArray<int>* searchsorted(NDArray<T>* a, NDArray<T>* v, std::string& side, NDArray<int>* sorter) {
		auto casted = broadcast({ a, v, sorter });

		a = (NDArray<T>*)casted.at(0);
		v = (NDArray<T>*)casted.at(1);
		sorter = (NDArray<int>*)casted.at(2);

		NDArray<int>* out = new NDArray<int>({ a->shape() });

		for (int i = 0; i < out->size(); i++) {
			out->at(i) = searchsorted(a, v->at(i), side, sorter);
		}

		return out;
	}

	template <typename T>
	NDArray<int>* searchsorted(NDArray<T>* a, NDArray<T>* v, std::string& side) {
		auto casted = broadcast({ a, v });

		a = (NDArray<T>*)casted.at(0);
		v = (NDArray<T>*)casted.at(1);

		NDArray<int>* out = new NDArray<int>({ v->size() });

		for (int i = 0; i < out->size(); i++) {
			out->at(i) = searchsorted(a, v->at(i), side);
		}

		return out;
	}

	template <typename T>
	inline NDArray<int>* searchsorted(NDArray<T>* a, NDArray<T>* v, NDArray<int>* sorter) {
		return searchsorted(a, v, "left", sorter);
	}

	template <typename T>
	NDArray<int>* searchsorted(NDArray<T>* a, NDArray<T>* v) {
		return searchsorted(a, v, "left");
	}

	template <typename T>
	inline NDArray<T>* extract(NDArray<bool>* condition, NDArray<T>* arr) {
		auto casted = broadcast({ condition, arr });

		condition = (NDArray<bool>*)casted.at(0);
		arr = (NDArray<T>*)casted.at(1);

		int size = 0;

		for (int i = 0; i < condition->size(); i++) {
			if (condition->at(i)) {
				size++;
			}
		}

		NDArray<T>* res = new NDArray<T>({ size });

		int index = 0;
		for (int i = 0; i < arr->size(); i++) {
			if (condition->at(i)) {
				res->at(index) = arr->at(i);
				index++;
			}
		}

		return res;
	}

	template <typename T>
	int count_nonzero(NDArray<T>* a) {
		int count = 0;

		for (int i = 0; i < a->size(); i++) {
			if (a->at(i) != 0) {
				count++;
			}
		}

		return count;
	}
}