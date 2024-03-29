#pragma once

#include "../Arrays/Arrays.cuh"

using namespace Cudheart::NDArrays;

namespace Cudheart::Logic {
	template <typename T>
	bool all(NDArray<T>* arr) {
		for (int i = 0; i < arr->size(); i++) {
			if (!(bool)arr->get(i)) {
				return false;
			}
		}

		return true;
	}

	template <typename T>
	bool any(NDArray<T>* arr) {
		for (int i = 0; i < arr->size(); i++) {
			if ((bool)arr->get(i)) {
				return true;
			}
		}

		return false;
	}

	template <typename T>
	NDArray<bool>* logicalAnd(NDArray<T>* a, NDArray<T>* b) {
		a->assertMatchShape(b->getShape());

		NDArray<bool>* out = a->emptyLike();

		for (int i = 0; i < a->size(); i++) {
			out->set(i, a->get(i) && b->get(i));
		}

		return out;
	}

	template <typename T>
	NDArray<bool>* logicalOr(NDArray<T>* a, NDArray<T>* b) {
		a->assertMatchShape(b->getShape());

		NDArray<bool>* out = a->emptyLike();

		for (int i = 0; i < a->size(); i++) {
			out->set(i, a->get(i) || b->get(i));
		}

		return out;
	}

	template <typename T>
	NDArray<bool>* logicalNot(NDArray<T>* arr) {
		NDArray<bool>* out = arr->emptyLike();

		for (int i = 0; i < arr->size(); i++) {
			out->set(i, !arr->get(i));
		}

		return out;
	}

	template <typename T>
	NDArray<bool>* logicalXor(NDArray<T>* a, NDArray<T>* b) {
		a->assertMatchShape(b->getShape());

		NDArray<bool>* out = a->emptyLike();

		for (int i = 0; i < a->size(); i++) {
			out->set(i, a->get(i) ^ b->get(i));
		}

		return out;
	}

	template <typename T>
	NDArray<bool>* isclose(NDArray<T>* a, NDArray<T>* b, double rtol, double atol) {
		a->assertMatchShape(b->getShape());

		auto out = new Vector<bool>(a->size());

		for (int i = 0; i < a->size(); i++) {
			T va = a->get(i);
			T vb = b->get(i);

			if (abs(va - vb) > (atol + rtol * abs(vb))) {
				out->set(i, false);
			}
			else {
				out->set(i, true);
			}
		}

		return out->reshape(a->getShape());
	}
	template <typename T>
	NDArray<bool>* isclose(NDArray<T>* a, NDArray<T>* b) {
		return isclose(a, b, 1e-05, 1e-08);
	}

	template <typename T>
	NDArray<bool>* isclose(NDArray<T>* a, T b, double rtol, double atol) {
		auto out = new Vector<bool>(a->size());

		for (int i = 0; i < a->size(); i++) {
			T va = a->get(i);

			if (abs(va - b) > (atol + rtol * abs(b))) {
				out->set(i, false);
			}
			else {
				out->set(i, true);
			}
		}

		return out->reshape(a->getShape());
	}

	template <typename T>
	NDArray<bool>* isclose(NDArray<T>* a, T b) {
		return isclose(a, b, 1e-05, 1e-08);
	}

	template <typename T>
	bool allclose(NDArray<T>* a, NDArray<T>* b, double rtol, double atol) {
		a->assertMatchShape(b->getShape());

		for (int i = 0; i < a->size(); i++) {
			T va = a->get(i);
			T vb = b->get(i);

			if (abs(va - vb) > (atol + rtol * abs(vb))) {
				return false;
			}
		}

		return true;
	}
	template <typename T>
	bool allclose(NDArray<T>* a, NDArray<T>* b) {
		return allclose(a, b, 1e-05, 1e-08);
	}

	template <typename T>
	bool allclose(NDArray<T>* a, T b, double rtol, double atol) {
		for (int i = 0; i < a->size(); i++) {
			T va = a->get(i);

			if (abs(va - b) > (atol + rtol * abs(b))) {
				return false;
			}
		}

		return true;
	}

	template <typename T>
	bool allclose(NDArray<T>* a, T b) {
		return allclose(a, b, 1e-05, 1e-08);
	}

	template<typename T>
	NDArray<bool>* equals(NDArray<T>* a, NDArray<T>* b) {
		return isclose(a, b, 0, 0);
	}

	template<typename T>
	NDArray<bool>* equals(NDArray<T>* a, T b) {
		return isclose(a, b, 0, 0);
	}

#pragma region greater
	template<typename T>
	NDArray<bool>* greater(NDArray<T>* a, NDArray<T>* b) {
		a->assertMatchShape(b->getShape());

		NDArray<bool>* out = a->emptyLike<bool>();

		for (int i = 0; i < a->size(); i++) {
			out->set(i, a->get(i) > b->get(i));
		}

		return out;
	}

	template<typename T>
	NDArray<bool>* greater(NDArray<T>* a, T b) {
		NDArray<bool>* out = a->emptyLike<bool>();

		for (int i = 0; i < a->size(); i++) {
			out->set(i, a->get(i) > b);
		}

		return out;
	}
#pragma endregion

#pragma region greaterEquals
	template<typename T>
	NDArray<bool>* greaterEquals(NDArray<T>* a, NDArray<T>* b) {
		a->assertMatchShape(b->getShape());

		NDArray<bool>* out = a->emptyLike<bool>();

		for (int i = 0; i < a->size(); i++) {
			out->set(i, a->get(i) >= b->get(i));
		}

		return out;
	}

	template<typename T>
	NDArray<bool>* greaterEquals(NDArray<T>* a, T b) {
		NDArray<bool>* out = a->emptyLike<bool>();

		for (int i = 0; i < a->size(); i++) {
			out->set(i, a->get(i) >= b);
		}

		return out;
	}
#pragma endregion

#pragma region less
	template<typename T>
	NDArray<bool>* less(NDArray<T>* a, NDArray<T>* b) {
		a->assertMatchShape(b->getShape());

		NDArray<bool>* out = a->emptyLike<bool>();

		for (int i = 0; i < a->size(); i++) {
			out->set(i, a->get(i) < b->get(i));
		}

		return out;
	}

	template<typename T>
	NDArray<bool>* less(NDArray<T>* a, T b) {
		NDArray<bool>* out = a->emptyLike<bool>();

		for (int i = 0; i < a->size(); i++) {
			out->set(i, a->get(i) < b);
		}

		return out;
	}
#pragma endregion

#pragma region lessEqual
	template<typename T>
	NDArray<bool>* lessEqual(NDArray<T>* a, NDArray<T>* b) {
		a->assertMatchShape(b->getShape());

		NDArray<bool>* out = a->emptyLike<bool>();

		for (int i = 0; i < a->size(); i++) {
			out->set(i, a->get(i) <= b->get(i));
		}

		return out;
	}

	template<typename T>
	NDArray<bool>* lessEqual(NDArray<T>* a, T b) {
		NDArray<bool>* out = a->emptyLike<bool>();

		for (int i = 0; i < a->size(); i++) {
			out->set(i, a->get(i) <= b);
		}

		return out;
	}
#pragma endregion
	template <typename T>
	NDArray<T>* maximum(NDArray<T>* a, NDArray<T>* b) {
		a->assertMatchShape(b->getShape());

		NDArray<T>* out = a->emptyLike();

		for (int i = 0; i < a->size(); i++) {
			out->set(i, std::max(a->get(i), b->get(i)));
		}

		return out;
	}

	template <typename T>
	T amax(NDArray<T>* x) {
		T max = x->get(0);

		for (int i = 1; i < x->size(); i++) {
			max = std::max(max, x->get(i));
		}

		return max;
	}

	template <typename T>
	NDArray<T>* minimum(NDArray<T>* a, NDArray<T>* b) {
		a->assertMatchShape(b->getShape());

		NDArray<T>* out = a->emptyLike();

		for (int i = 0; i < a->size(); i++) {
			out->set(i, std::min(a->get(i), b->get(i)));
		}

		return out;
	}

	template <typename T>
	T amin(NDArray<T>* x) {
		T min = x->get(0);

		for (int i = 1; i < x->size(); i++) {
			min = std::min(min, x->get(i));
		}

		return min;
	}
}