#pragma once

#include "../../Arrays/NDArray.cuh"
#include "../../Internal/Internal.cuh"
#include <cmath>

namespace CudheartNew::CPP::Math::Exp {
	template <typename T>
	NDArray<T>* ln(NDArray<T>* x) {
		NDArray<T>* output = new NDArray<T>(x->shape());

		for (int i = 0; i < x->size(); i++) {
			output->at(i) = log(x->at(i));
		}

		return output;
	}

	template <typename T>
	NDArray<T>* loga2(NDArray<T>* x) {
		NDArray<T>* output = new NDArray<T>(x->shape());

		for (int i = 0; i < x->size(); i++) {
			output->at(i) = log2(x->at(i));
		}

		return output;
	}

	template <typename A, typename B, typename T = promote(A, B)>
	NDArray<T>* logan(NDArray<A>* x, B n) {
		NDArray<T>* output = new NDArray<T>(x->shape());

		for (int i = 0; i < x->size(); i++) {
			// using the change of bases rule:
			// log n of x[i] = log(x[i]) / log(n)
			output->at(i) = (log(x->at(i)) / log(n));
		}

		return output;
	}

	template <typename A, typename B, typename T = promote(A, B)>
	NDArray<T>* logan(NDArray<A>* x, NDArray<B>* n) {
		auto casted = broadcast({ x, n });

		x = (NDArray<A>*)casted[0];
		n = (NDArray<B>*)casted[1];

		NDArray<T>* output = new NDArray<T>(x->shape());

		for (int i = 0; i < x->size(); i++) {
			// using the change of bases rule:
			// log n of x[i] = log(x[i]) / log(n)
			output->at(i) = (log(x->at(i)) / log(n->at(i)));
		}

		return output;
	}

	template <typename T>
	NDArray<T>* loga10(NDArray<T>* x) {
		NDArray<T>* output = new NDArray<T>(x->shape());

		for (int i = 0; i < x->size(); i++) {
			output->at(i) = log10(x->at(i));
		}

		return output;
	}

	template <typename T>
	NDArray<T>* expo(NDArray<T>* x) {
		NDArray<T>* output = new NDArray<T>(x->shape());

		for (int i = 0; i < x->size(); i++) {
			output->at(i) = std::exp(x->at(i));
		}

		return output;
	}

	template <typename T>
	NDArray<T>* expom1(NDArray<T>* x) {
		NDArray<T>* output = new NDArray<T>(x->shape());

		for (int i = 0; i < x->size(); i++) {
			output->at(i) = std::expm1(x->at(i));
		}

		return output;
	}

	template <typename T>
	NDArray<T>* expo2(NDArray<T>* x) {
		NDArray<T>* output = new NDArray<T>(x->shape());

		for (int i = 0; i < x->size(); i++) {
			output->at(i) = std::exp2(x->at(i));
		}

		return output;
	}

	template <typename A, typename B, typename T = promote(A, B)>
	NDArray<T>* logaddexp(NDArray<A>* a, NDArray<B>* b) {
		auto casted = broadcast({ a, b });

		a = (NDArray<A>*)casted[0];
		b = (NDArray<B>*)casted[1];

		NDArray<T>* output = new NDArray<T>(a->shape());

		for (int i = 0; i < a->size(); i++) {
			output->at(i) = std::log(std::exp(a->at(i)) + std::exp(b->at(i)));
		}

		return output;
	}

	template <typename A, typename B, typename T = promote(A, B)>
	NDArray<T>* logaddexp2(NDArray<A>* a, NDArray<B>* b) {
		auto casted = broadcast({ a, b });

		a = (NDArray<A>*)casted[0];
		b = (NDArray<B>*)casted[1];

		NDArray<T>* output = new NDArray<T>(a->shape());

		for (int i = 0; i < a->size(); i++) {
			output->at(i) = std::log2(std::exp2(a->at(i)) + std::exp2(b->at(i)));
		}

		return output;
	}
}