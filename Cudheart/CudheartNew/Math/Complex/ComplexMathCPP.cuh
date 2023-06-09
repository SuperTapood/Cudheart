#pragma once

#include "../../Arrays/NDArray.cuh"
#include <cmath>
#include <numeric>
#include <complex>
#include "../Constants.cuh"

namespace CudheartNew::CPP::Math::Complex {
	template <typename T>
	NDArray<std::complex<T>>* angle(NDArray<std::complex<T>>* z, bool deg = false) {
		auto arr = new NDArray<std::complex<T>>(z->shape());
		double multiplier = 1.0;

		if (deg) {
			multiplier = 180 / pi;
		}

		for (int i = 0; i < z->size(); i++) {
			auto current = z->at(i);
			arr->at(i) = std::arg(current) * multiplier;
		}

		return arr;
	}

	template <typename T>
	NDArray<std::complex<T>>* real(NDArray<std::complex<T>*>* val) {
		auto arr = new NDArray<std::complex<T>>(val->shape());

		for (int i = 0; i < val->size(); i++) {
			std::complex<T>* current = val->at(i);
			arr->at(i) = current->real;
		}

		return arr;
	}

	template <typename T>
	NDArray<std::complex<T>>* imag(NDArray<std::complex<T>*>* val) {
		auto arr = new NDArray<std::complex<T>>(val->shape());

		for (int i = 0; i < val->size(); i++) {
			std::complex<T>* current = val->at(i);
			arr->at(i) = current->imag;
		}

		return arr;
	}

	template <typename T>
	NDArray<std::complex<T>*>* conj(NDArray<std::complex<T>*>* x) {
		auto arr = new NDArray<std::complex<T>>(x->shape());

		for (int i = 0; i < x->size(); i++) {
			std::complex<T>* current = x->at(i);
			arr->at(i) = std::complex<T>(current->real, -current->imag);
		}

		return arr;
	}

	template <typename T>
	NDArray<std::complex<T>*>* abs(NDArray<std::complex<T>*>* x) {
		auto arr = new NDArray<std::complex<T>>(x->shape());

		for (int i = 0; i < x->size(); i++) {
			std::complex<T>* current = x->at(i);
			arr->at(i) = std::abs(current);
		}

		return arr;
	}

	template <typename T>
	NDArray<std::complex<T>*>* complexSign(NDArray<std::complex<T>*>* x) {
		auto out = new NDArray<std::complex<T>*>(x->shape());

		for (int i = 0; i < x->size(); i++) {
			double real = x->at(i)->real;
			if (std::signbit(real)) {
				out->at(i) = std::complex<T>(-1, 0);
			}
			else {
				out->at(i) = std::complex<T>(1, 0);
			}
		}

		return out;
	}
}