#pragma once

#include "../../Arrays/NDArray.cuh"

#include "../../Internal/Internal.cuh"

namespace CudheartNew::CPP::Math::Bitwise {
	template <typename A, typename B, typename T = promote(A, B)>
	NDArray<T>* bitwiseAnd(NDArray<A>* a, NDArray<B>* b) {
		auto casted = broadcast({ a, b });

		a = (NDArray<A>*)casted[0];
		b = (NDArray<B>*)casted[1];

		NDArray<T>* out = new NDArray<T>(a->shape());

		for (int i = 0; i < a->size(); i++) {
			out->at(i) = a->at(i) & b->at(i);
		}

		return out;
	}

	template <typename A, typename B, typename T = promote(A, B)>
	NDArray<T>* bitwiseOr(NDArray<A>* a, NDArray<B>* b) {
		auto casted = broadcast({ a, b });

		a = (NDArray<A>*)casted[0];
		b = (NDArray<B>*)casted[1];

		NDArray<T>* out = new NDArray<T>(a->shape());

		for (int i = 0; i < a->size(); i++) {
			out->at(i) = a->at(i) | b->at(i);
		}

		return out;
	}

	template <typename A, typename B, typename T = promote(A, B)>
	NDArray<T>* bitwiseXor(NDArray<A>* a, NDArray<B>* b) {
		auto casted = broadcast({ a, b });

		a = (NDArray<A>*)casted[0];
		b = (NDArray<B>*)casted[1];

		NDArray<T>* out = new NDArray<T>(a->shape());

		for (int i = 0; i < a->size(); i++) {
			out->at(i) = a->at(i) ^ b->at(i);
		}

		return out;
	}

	template <typename A, typename B, typename T = promote(A, B)>
	NDArray<T>* bitwiseLeftShift(NDArray<A>* a, NDArray<B>* b) {
		auto casted = broadcast({ a, b });

		a = (NDArray<A>*)casted[0];
		b = (NDArray<B>*)casted[1];

		NDArray<T>* out = new NDArray<T>(a->shape());

		for (int i = 0; i < a->size(); i++) {
			out->at(i) = a->at(i) << b->at(i);
		}

		return out;
	}

	template <typename A, typename B, typename T = promote(A, B)>
	NDArray<T>* bitwiseRightShift(NDArray<A>* a, NDArray<B>* b) {
		auto casted = broadcast({ a, b });

		a = (NDArray<A>*)casted[0];
		b = (NDArray<B>*)casted[1];

		NDArray<T>* out = new NDArray<T>(a->shape());

		for (int i = 0; i < a->size(); i++) {
			out->at(i) = a->at(i) >> b->at(i);
		}

		return out;
	}

	template <typename T>
	NDArray<T>* bitwiseNot(NDArray<T>* x) {
		NDArray<T>* out = new NDArray<T>(x->shape());

		for (int i = 0; i < x->size(); i++) {
			out->at(i) = ~(x->at(i));
		}

		return out;
	}
}