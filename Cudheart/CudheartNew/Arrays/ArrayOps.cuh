#pragma once

#include "NDArray.cuh"

#define FMT_HEADER_ONLY
#include "fmt/core.h"

namespace Cudheart::ArrayOps {
	template <typename T>
	NDArray<T>* empty(std::vector<int> shape) {
		return new NDArray<T>(shape);
	}

	template <typename T>
	NDArray<T>* emptyLike(NDArray<T>* arr) {
		return new NDArray<T>(arr->shape());
	}

	template <typename A, typename B>
	NDArray<B>* emptyLike(NDArray<A>* arr) {
		return new NDArray<B>(arr->shape());
	}

	template <typename T>
	NDArray<T>* arange(T start, T end, T jump) {
		if (signbit((double)(end - start)) != signbit((double)jump)) {
			fmt::println("cannot define range from {} to {}", start, end);
			exit(1);
		}

		// thanks c++
		float diff = (float)end - (float)start;
		float steps = diff / (float)jump;
		auto len = (int)ceil(steps);

		NDArray<T>* out = new NDArray<T>(len);

		for (int i = 0; i < len; start += jump, i++) {
			out->at(i) = start;
		}

		return out;
	}

	template <typename T>
	NDArray<T>* arange(T end, T jump) {
		return arange((T)0, end, jump);
	}

	template <typename T>
	NDArray<T>* arange(T end) {
		return arange((T)0, end, (T)1);
	}

	template <typename T>
	NDArray<T>* full(std::vector<int> shape, T value) {
		NDArray<T>* out = new NDArray<T>(shape);

		for (int i = 0; i < out->size(); i++) {
			out->at(i) = value;
		}

		return out;
	}

	template <typename T>
	NDArray<T>* fullLike(NDArray<T>* arr, T value) {
		return fullLike(arr->shape(), value);
	}

	template <typename T>
	NDArray<T>* linspace(T start, T stop, int num, bool endpoint) {
		double diff = (double)stop - (double)start;

		NDArray<T>* out = new NDArray<T>({ num });

		if (endpoint) {
			num--;
			out->at(-1) = stop;
		}

		T jump = (T)(diff / num);

		for (int i = 0; i < num; i++) {
			out->at(i) = (T)(start + jump * i);
		}

		return out;
	}

	template <typename T>
	NDArray<T>* linspace(T start, T stop, int num) {
		return linspace<T>(start, stop, num, true);
	}

	template <typename T>
	NDArray<T>* linspace(T start, T stop) {
		return linspace<T>(start, stop, 50, true);
	}

	template <typename T>
	NDArray<T>* ones(std::vector<int> shape) {
		return full(shape, 1);
	}

	template <typename T>
	NDArray<T>* onesLike(NDArray<T>* arr) {
		return ones(arr->shape());
	}

	template <typename T>
	NDArray<T>* zeros(std::vector<int> shape) {
		return full(shape, 0);
	}

	template <typename T>
	NDArray<T>* zerosLike(NDArray<T>* arr) {
		return zeros(arr->shape());
	}

	template <typename T>
	NDArray<T>* logspace(T start, T stop, int num, bool endpoint, T base) {
		double diff = (double)stop - (double)start;

		NDArray<T>* out = new NDArray<T>({num});

		if (endpoint) {
			num--;
			out->at(-1) = pow(base, stop);
		}

		T jump = (T)(diff / num);

		for (int i = 0; i < num; i++) {
			out->at(i) = pow(base, (T)(start + jump * i));
		}

		return out;
	}

	template <typename T>
	NDArray<T>* logspace(T start, T stop, int num, bool endpoint) {
		return logspace(start, stop, num, endpoint, 10.0);
	}

	template <typename T>
	NDArray<T>* logspace(T start, T stop, int num, double base) {
		return logspace(start, stop, num, true, base);
	}

	template <typename T>
	NDArray<T>* logspace(T start, T stop, int num) {
		return logspace(start, stop, num, true, 10.0);
	}

	template <typename T>
	NDArray<T>* logspace(T start, T stop) {
		return logspace(start, stop, 50, true, 10.0);
	}

	template <typename T>
	NDArray<T>* geomspace(T start, T stop, int num, bool endpoint) {
		return logspace((T)log10(start), (T)log10(stop), num, endpoint, (T)10.0);
	}

	template <typename T>
	NDArray<T>* geomspace(T start, T stop, int num) {
		return geomspace<T>(start, stop, num, true);
	}

	template <typename T>
	NDArray<T>* geomspace(T start, T stop) {
		return geomspace<T>(start, stop, (T)50, true);
	}
}