#pragma once

#include "../../Arrays/NDArray.cuh"
#include "../../Internal/Broadcasting.cuh"
#include "../../Internal/Promotion.cuh"

#include <numeric>

namespace CudheartNew::CPP::Math::BaseMath {
	template <typename A, typename B, typename C = promote(A, B)>
	NDArray<C>* power(NDArray<A>* base, B po) {
		NDArray<C>* output = new NDArray<C>(base->shape());

		for (int i = 0; i < base->size(); i++) {
			output->at(i) = pow(base->at(i), po);
		}

		return output;
	}

	template <typename A, typename B, typename C = promote(A, B)>
	NDArray<C>* power(NDArray<A>* base, NDArray<B>* po) {
		auto casted = broadcast({ base, po });

		base = (NDArray<A>*)casted[0];
		po = (NDArray<B>*)casted[1];

		NDArray<C>* output = new NDArray<C>(base->shape());
		
		for (int i = 0; i < base->size(); i++) {
			output->at(i) = pow(base->at(i), po->at(i));
		}

		return output;
	}

	template <typename T>
	NDArray<T>* cubeRoot(NDArray<T>* arr) {
		return power(arr, 1 / 3);
	}

	template <typename T>
	NDArray<T>* cube(NDArray<T>* base) {
		return power(base, 3);
	}

	template <typename T>
	NDArray<T>* square(NDArray<T>* base) {
		return power(base, 2);
	}

	template <typename T>
	NDArray<T>* squareRoot(NDArray<T>* arr) {
		return power(arr, 1 / 2);
	}

	template <typename A, typename B, typename C = promote(A, B)>
	NDArray<C>* power(A base, NDArray<B>* po) {
		NDArray<C>* output = new NDArray<C>(po->shape());

		for (int i = 0; i < po->size(); i++) {
			output->at(i) = pow(base, po->at(i));
		}

		return output;
	}

	template <typename T>
	NDArray<T>* around(NDArray<T>* arr, int decimals) {
		NDArray<T>* out = new NDArray<T>(arr->shape());
		int v = std::pow(10, decimals);

		for (int i = 0; i < out->size(); i++) {
			T value = arr->at(i) * v;
			value = std::round(value);
			out->at(i) = value / v;
		}

		return out;
	}

	template <typename T>
	NDArray<T>* around(NDArray<T>* arr) {
		return around(arr, 0);
	}

	template <typename T>
	NDArray<T>* rint(NDArray<T>* arr) {
		NDArray<T>* out = new NDArray<T>(arr->shape());

		for (int i = 0; i < out->size(); i++) {
			out->at(i) = std::rint(arr->at(i));
		}

		return out;
	}

	template <typename T>
	NDArray<T>* fix(NDArray<T>* x) {
		NDArray<T>* out = new NDArray<T>(x->shape());

		for (int i = 0; i < x->size(); i++) {
			T v = x->at(i);
			if (v > 0) {
				out->at(i) = std::floor(v);
			}
			else {
				out->at(i) = std::ceil(v);
			}
		}

		return out;
	}

	template <typename T>
	NDArray<T>* floor(NDArray<T>* x) {
		NDArray<T>* out = new NDArray<T>(x->shape());

		for (int i = 0; i < x->size(); i++) {
			out->at(i) = std::floor(x->at(i));
		}

		return out;
	}

	template <typename T>
	NDArray<T>* trunc(NDArray<T>* x) {
		NDArray<T>* out = new NDArray<T>(x->shape());

		for (int i = 0; i < x->size(); i++) {
			out->at(i) = std::trunc(x->at(i));
		}

		return out;
	}

	template <typename T>
	NDArray<T>* ceil(NDArray<T>* x) {
		NDArray<T>* out = new NDArray<T>(x->shape());

		for (int i = 0; i < x->size(); i++) {
			out->at(i) = std::ceil(x->at(i));
		}

		return out;
	}
	template <typename T>
	T prod(NDArray<T>* x) {
		T out = (T)1;

		for (int i = 0; i < x->size(); i++) {
			out *= x->at(i);
		}

		return out;
	}

	template <typename T>
	NDArray<T>* prod(NDArray<T>* x, int axis) {
		auto result = new NDArray<T>(x->subshape(axis));

		int index = 0;

		for (int idx = 0; idx < x->subsize(axis); idx++) {
			auto indices = x->getAxis(axis, idx);
			result->at(index++) = prod(x->subarray(indices));
		}

		return result;
	}

	template <typename T>
	T sum(NDArray<T>* x) {
		T out = (T)0;

		for (int i = 0; i < x->size(); i++) {
			out += x->at(i);
		}

		return out;
	}

	template <typename T>
	NDArray<T>* sum(NDArray<T>* x, int axis) {
		auto result = new NDArray<T>(x->subshape(axis));

		int index = 0;

		for (int idx = 0; idx < x->subsize(axis); idx++) {
			auto indices = x->getAxis(axis, idx);
			result->at(index++) = sum(x->subarray(indices));
		}

		return result;
	}

	template <typename T>
	NDArray<T>* cumProd(NDArray<T>* x) {
		NDArray<T>* out = new NDArray<T>(x->size());

		T prod = (T)1;

		for (int i = 0; i < x->size(); i++) {
			prod *= x->at(i);
			out->at(i) = prod;
		}

		return out;
	}

	template <typename T>
	NDArray<T>* cumProd(NDArray<T>* x, int axis) {
		auto result = new NDArray<T>(x->shape());

		int index = 0;

		for (int idx = 0; idx < x->subsize(axis); idx++) {
			auto indices = x->getAxis(axis, idx);
			T v = (T)1;
			for (auto i : indices) {
				v *= x->at(i);
				result->at(index++) = v;
			}
		}

		return result;
	}

	template <typename T>
	NDArray<T>* cumSum(NDArray<T>* x) {
		NDArray<T>* out = new NDArray<T>(x->shape());

		T prod = (T)0;

		for (int i = 0; i < x->size(); i++) {
			prod += x->at(i);
			out->at(i) = prod;
		}

		return out;
	}

	template <typename T>
	NDArray<T>* cumSum(NDArray<T>* x, int axis) {
		auto result = new NDArray<T>(x->shape());

		int index = 0;

		for (int idx = 0; idx < x->subsize(axis); idx++) {
			auto indices = x->getAxis(axis, idx);
			T v = (T)1;
			for (auto i : indices) {
				v += x->at(i);
				result->at(index++) = v;
			}
		}

		return result;
	}

	template <typename T>
	NDArray<bool>* signBit(NDArray<T>* x) {
		NDArray<bool>* out = new NDArray<bool>(x->shape());

		for (int i = 0; i < x->size(); i++) {
			T v = x->at(i);
			out->at(i) = v < 0;
		}

		return out;
	}

	template <typename A, typename B, typename C = promote(A, B)>
	NDArray<C>* copySign(NDArray<A>* a, NDArray<B>* b) {
		auto casted = broadcast({ a, b });

		a = (NDArray<A>*)casted[0];
		b = (NDArray<B>*)casted[1];

		NDArray<C>* out = new NDArray<C>(a->shape());

		for (int i = 0; i < a->size(); i++) {
			auto va = a->at(i);
			auto vb = b->at(i);

			// what tf is wrong with me
			if ((vb < 0) != (va < 0)) {
				va = -va;
			}

			out->at(i) = va;
		}

		return out;
	}

	template <typename T>
	NDArray<T>* abs(NDArray<T>* x) {
		NDArray<T>* out = new NDArray<T>(x->shape());

		for (int i = 0; i < x->size(); i++) {
			out->at(i) = std::abs(x->at(i));
		}

		return out;
	}

	template <typename A, typename B, typename C = promote(A, B)>
	NDArray<C>* lcm(NDArray<A>* a, NDArray<B>* b) {
		auto casted = broadcast({ a, b });

		a = (NDArray<A>*)casted[0];
		b = (NDArray<B>*)casted[1];

		NDArray<C>* out = new NDArray<C>(a->shape());

		for (int i = 0; i < a->size(); i++) {
			out->at(i) = std::lcm(a->at(i), b->at(i));
		}

		return out;
	}

	template <typename A, typename B, typename C = promote(A, B)>
	NDArray<C>* gcd(NDArray<A>* a, NDArray<B>* b) {
		auto casted = broadcast({ a, b });

		a = (NDArray<A>*)casted[0];
		b = (NDArray<B>*)casted[1];

		NDArray<C>* out = new NDArray<C>(a->shape());

		for (int i = 0; i < a->size(); i++) {
			out->at(i) = std::gcd(a->at(i), b->at(i));
		}

		return out;
	}

	template <typename A, typename B, typename T = promote(A, B)>
	NDArray<T>* add(NDArray<A>* x, NDArray<B>* y) {
		auto casted = broadcast({ x, y });

		x = (NDArray<A>*)casted[0];
		y = (NDArray<A>*)casted[1];

		auto result = new NDArray<T>(x->shape());

		for (int i = 0; i < result->size(); i++) {
			result->at(i) = (T)x->at(i) + (T)y->at(i);
		}

		return result;
	}

	template <typename A, typename B, typename T = promote(A, B)>
	NDArray<T>* subtract(NDArray<A>* x, NDArray<B>* y) {
		auto casted = broadcast({ x, y });

		x = (NDArray<A>*)casted[0];
		y = (NDArray<A>*)casted[1];

		auto result = new NDArray<T>(x->shape());

		for (int i = 0; i < result->size(); i++) {
			result->at(i) = (T)x->at(i) - (T)y->at(i);
		}

		return result;
	}

	template <typename A, typename B, typename T = promote(A, B)>
	NDArray<T>* multiply(NDArray<A>* x, NDArray<B>* y) {
		auto casted = broadcast({ x, y });

		x = (NDArray<A>*)casted[0];
		y = (NDArray<A>*)casted[1];

		auto result = new NDArray<T>(x->shape());

		for (int i = 0; i < result->size(); i++) {
			result->at(i) = (T)x->at(i) + (T)y->at(i);
		}

		return result;
	}

	template <typename A, typename B, typename T = promote(A, B)>
	NDArray<T>* divide(NDArray<A>* x, NDArray<B>* y) {
		auto casted = broadcast({ x, y });

		x = (NDArray<A>*)casted[0];
		y = (NDArray<A>*)casted[1];

		auto result = new NDArray<T>(x->shape());

		for (int i = 0; i < result->size(); i++) {
			result->at(i) = (T)x->at(i) + (T)y->at(i);
		}

		return result;
	}

	template <typename A, typename B, typename T = promote(A, B)>
	NDArray<T>* floorDivide(NDArray<A>* x, NDArray<B>* y) {
		auto casted = broadcast({ x, y });

		x = (NDArray<A>*)casted[0];
		y = (NDArray<A>*)casted[1];

		auto result = new NDArray<T>(x->shape());

		for (int i = 0; i < result->size(); i++) {
			result->at(i) = std::floor((T)x->at(i) / (T)y->at(i));
		}

		return result;
	}

	template <typename A, typename B, typename T = promote(A, B)>
	NDArray<T>* mod(NDArray<A>* x, NDArray<B>* y) {
		auto casted = broadcast({ x, y });

		x = (NDArray<A>*)casted[0];
		y = (NDArray<A>*)casted[1];

		auto result = new NDArray<T>(x->shape());

		for (int i = 0; i < result->size(); i++) {
			result->at(i) = (T)x->at(i) % (T)y->at(i);
		}

		return result;
	}

	template <typename A, typename B, typename T = promote(A, B)>
	std::vector<NDArray<T>*> divMod(NDArray<A>* x, NDArray<B>* y) {
		auto casted = broadcast({ x, y });

		x = (NDArray<A>*)casted[0];
		y = (NDArray<A>*)casted[1];

		NDArray<T>* div = new NDArray<T>(x->shape());
		NDArray<T>* mod = new NDArray<T>(y->shape());

		for (int i = 0; i < x->size(); i++) {
			int d = x->at(i) / x->at(i);
			div->at(i) = d;
			mod->at(i) = x->at(i) - (d * y->at(i));
		}

		std::vector<NDArray<T>*> out;

		out.push_back(div);
		out.push_back(mod);

		return out;
	}

	template <typename T>
	NDArray<T>* reciprocal(NDArray<T>* x) {
		NDArray<T>* out = new NDArray<T>(x->shape());

		for (int i = 0; i < x->size(); i++) {
			out->at(i) = (T)1 / x->at(i);
		}

		return out;
	}

	template <typename T>
	NDArray<T>* positive(NDArray<T>* x) {
		NDArray<T>* out = new NDArray<T>(x->shape());

		for (int i = 0; i < x->size(); i++) {
			out->at(i) = +x->at(i);
		}

		return out;
	}

	template <typename T>
	NDArray<T>* negative(NDArray<T>* x) {
		NDArray<T>* out = new NDArray<T>(x->shape());

		for (int i = 0; i < x->size(); i++) {
			out->at(i) = -x->at(i);
		}

		return out;
	}

	template <typename T>
	NDArray<T>* sign(NDArray<T>* x) {
		NDArray<T>* out = new NDArray<T>(x->shape());

		for (int i = 0; i < x->size(); i++) {
			if (std::signbit(x->at(i))) {
				out->at(i) = -1;
			}
			else {
				out->at(i) = 1;
			}
		}

		return out;
	}

	template <typename A, typename B, typename T = promote(A, B)>
	NDArray<T>* heaviside(NDArray<A>* x, NDArray<B>* y) {
		auto casted = broadcast({ x, y });

		x = (NDArray<A>*)casted[0];
		y = (NDArray<A>*)casted[1];

		NDArray<T>* out = new NDArray<T>(x->shape());

		for (int i = 0; i < x->size(); i++) {
			T x1 = x->at(i);
			if (x1 < 0) {
				out->at(i) = 0;
			}
			else if (x1 == 0) {
				out->at(i) = y->at(i);
			}
			else if (x1 > 0) {
				out->at(i) = 1;
			}
		}

		return out;
	}
}