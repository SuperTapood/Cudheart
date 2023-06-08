#pragma once

#include "../Arrays/NDArray.cuh"
#include "../Internal/Broadcasting.cuh"
#include "../Internal/Promotion.cuh"

namespace CudheartNew::Logic::CPP {
	template <typename T>
	bool all(NDArray<T>* arr) {
		for (int i = 0; i < arr->size(); i++) {
			if (!(bool)arr->at(i)) {
				return false;
			}
		}

		return true;
	}

	template <typename T>
	bool any(NDArray<T>* arr) {
		for (int i = 0; i < arr->size(); i++) {
			if ((bool)arr->at(i)) {
				return true;
			}
		}

		return false;
	}

	template <typename A, typename B>
	NDArray<bool>* logicalAnd(NDArray<A>* a, NDArray<B>* b) {
		auto casted = broadcast({ a, b });

		a = (NDArray<A>*)casted[0];
		b = (NDArray<B>*)casted[0];

		NDArray<bool>* out = new NDArray<bool>(a->shape());

		for (int i = 0; i < a->size(); i++) {
			out->at(i) = a->at(i) && b->at(i);
		}

		return out;
	}

	template <typename A, typename B>
	NDArray<bool>* logicalOr(NDArray<A>* a, NDArray<B>* b) {
		auto casted = broadcast({ a, b });

		a = (NDArray<A>*)casted[0];
		b = (NDArray<B>*)casted[0];

		NDArray<bool>* out = new NDArray<bool>(a->shape());

		for (int i = 0; i < a->size(); i++) {
			out->at(i) = a->at(i) || b->at(i);
		}

		return out;
	}

	template <typename T>
	NDArray<bool>* logicalNot(NDArray<T>* arr) {
		NDArray<bool>* out = new NDArray<bool>(arr->shape());

		for (int i = 0; i < arr->size(); i++) {
			out->at(i) = !arr->at(i);
		}

		return out;
	}

	template <typename A, typename B>
	NDArray<bool>* logicalXor(NDArray<A>* a, NDArray<B>* b) {
		auto casted = broadcast({ a, b });

		a = (NDArray<A>*)casted[0];
		b = (NDArray<B>*)casted[0];

		NDArray<bool>* out = new NDArray<bool>(a->shape());

		for (int i = 0; i < a->size(); i++) {
			out->at(i) = a->at(i) ^ b->at(i);
		}

		return out;
	}

	template <typename A, typename B>
	NDArray<bool>* isclose(NDArray<A>* a, NDArray<B>* b, double rtol, double atol) {
		auto casted = broadcast({ a, b });

		a = (NDArray<A>*)a;
		b = (NDArray<B>*)b;

		auto out = new NDArray<bool>(a->shape());

		for (int i = 0; i < a->size(); i++) {
			A va = a->at(i);
			B vb = b->at(i);

			if (std::abs(va - vb) > (atol + rtol * std::abs(vb))) {
				out->at(i) = false;
			}
			else {
				out->at(i) = true;
			}
		}

		return out;
	}

	template <typename A, typename B>
	NDArray<bool>* isclose(NDArray<A>* a, NDArray<B>* b) {
		return isclose(a, b, 1e-05, 1e-08);
	}

	template <typename A, typename B>
	NDArray<bool>* isclose(NDArray<A>* a, B b, double rtol, double atol) {
		auto out = new NDArray<bool>(a->shape());

		for (int i = 0; i < a->size(); i++) {
			A va = a->at(i);

			if (abs(va - b) > (atol + rtol * abs(b))) {
				out->at(i) = false;
			}
			else {
				out->at(i) = true;
			}
		}

		return out;
	}

	template <typename A, typename B>
	NDArray<bool>* isclose(NDArray<A>* a, B b) {
		return isclose(a, b, 1e-05, 1e-08);
	}

	template <typename A, typename B>
	bool allclose(NDArray<A>* a, NDArray<B>* b, double rtol, double atol) {
		auto casted = broadcast({ a, b });

		a = (NDArray<A>*)a;
		b = (NDArray<B>*)b;

		for (int i = 0; i < a->size(); i++) {
			A va = a->at(i);
			B vb = b->at(i);

			if (abs(va - vb) > (atol + rtol * abs(vb))) {
				return false;
			}
		}

		return true;
	}

	template <typename A, typename B>
	bool allclose(NDArray<A>* a, NDArray<B>* b) {
		return allclose(a, b, 1e-05, 1e-08);
	}

	template <typename A, typename B>
	bool allclose(NDArray<A>* a, B b, double rtol, double atol) {
		for (int i = 0; i < a->size(); i++) {
			A va = a->at(i);

			if (abs(va - b) > (atol + rtol * abs(b))) {
				return false;
			}
		}

		return true;
	}

	template <typename A, typename B>
	bool allclose(NDArray<A>* a, B b) {
		return allclose(a, b, 1e-05, 1e-08);
	}

	template <typename A, typename B>
	NDArray<bool>* equals(NDArray<A>* a, NDArray<B>* b) {
		return isclose(a, b, 0, 0);
	}

	template <typename A, typename B>
	NDArray<bool>* equals(NDArray<A>* a, B b) {
		return isclose(a, b, 0, 0);
	}

	template <typename A, typename B>
	NDArray<bool>* greater(NDArray<A>* a, NDArray<B>* b) {
		auto casted = broadcast({ a, b });

		a = (NDArray<A>*)a;
		b = (NDArray<B>*)b;

		NDArray<bool>* out = new NDArray<bool>(a->shape());

		for (int i = 0; i < a->size(); i++) {
			out->at(i) = a->at(i) > b->at(i);
		}

		return out;
	}

	template <typename A, typename B>
	NDArray<bool>* greater(NDArray<A>* a, B b) {
		NDArray<bool>* out = new NDArray<bool>(a->shape());

		for (int i = 0; i < a->size(); i++) {
			out->at(i) = a->at(i) > b;
		}

		return out;
	}

	template <typename A, typename B>
	NDArray<bool>* greaterEquals(NDArray<A>* a, NDArray<B>* b) {
		auto casted = broadcast({ a, b });

		a = (NDArray<A>*)a;
		b = (NDArray<B>*)b;

		NDArray<bool>* out = new NDArray<bool>(a->shape());

		for (int i = 0; i < a->size(); i++) {
			out->at(i) = a->at(i) >= b->at(i);
		}

		return out;
	}

	template <typename A, typename B>
	NDArray<bool>* greaterEquals(NDArray<A>* a, B b) {
		NDArray<bool>* out = new NDArray<bool>(a->shape());

		for (int i = 0; i < a->size(); i++) {
			out->at(i) = a->at(i) >= b;
		}

		return out;
	}

	template <typename A, typename B>
	NDArray<bool>* less(NDArray<A>* a, NDArray<B>* b) {
		auto casted = broadcast({ a, b });

		a = (NDArray<A>*)a;
		b = (NDArray<B>*)b;

		NDArray<bool>* out = new NDArray<bool>(a->shape());

		for (int i = 0; i < a->size(); i++) {
			out->at(i) = a->at(i) < b->at(i);
		}

		return out;
	}

	template <typename A, typename B>
	NDArray<bool>* less(NDArray<A>* a, B b) {
		NDArray<bool>* out = new NDArray<bool>(a->shape());

		for (int i = 0; i < a->size(); i++) {
			out->at(i) = a->at(i) < b;
		}

		return out;
	}
	
	template <typename A, typename B>
	NDArray<bool>* lessEqual(NDArray<A>* a, NDArray<B>* b) {
		auto casted = broadcast({ a, b });

		a = (NDArray<A>*)a;
		b = (NDArray<B>*)b;

		NDArray<bool>* out = new NDArray<bool>(a->shape());

		for (int i = 0; i < a->size(); i++) {
			out->at(i) = a->at(i) <= b->at(i);
		}

		return out;
	}

	template <typename A, typename B>
	NDArray<bool>* lessEqual(NDArray<A>* a, B b) {
		NDArray<bool>* out = new NDArray<bool>(a->shape());

		for (int i = 0; i < a->size(); i++) {
			out->at(i) = a->at(i) <= b;
		}

		return out;
	}

	template <typename A, typename B, typename C = promote(A, B)>
	NDArray<C>* maximum(NDArray<A>* a, NDArray<B>* b) {
		auto casted = broadcast({ a, b });

		a = (NDArray<A>*)a;
		b = (NDArray<B>*)b;

		NDArray<C>* out = new NDArray<C>(a->shape());

		for (int i = 0; i < a->size(); i++) {
			out->at(i) = std::max(a->at(i), b->at(i));
		}

		return out;
	}

	template <typename T>
	T amax(NDArray<T>* x) {
		T max = x->at(0);

		for (int i = 1; i < x->size(); i++) {
			max = std::max(max, x->at(i));
		}

		return max;
	}

	template <typename A, typename B, typename C = promote(A, B)>
	NDArray<C>* minimum(NDArray<A>* a, NDArray<B>* b) {
		auto casted = broadcast({a, b});

		a = (NDArray<A>*)a;
		b = (NDArray<B>*)b;

		NDArray<C>* out = new NDArray<C>(a->shape());

		for (int i = 0; i < a->size(); i++) {
			out->at(i) = std::min(a->at(i), b->at(i));
		}

		return out;
	}

	template <typename T>
	T amin(NDArray<T>* x) {
		T min = x->at(0);

		for (int i = 1; i < x->size(); i++) {
			min = std::min(min, x->at(i));
		}

		return min;
	}
}