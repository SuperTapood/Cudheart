#pragma once

#include "../Util.cuh"
#include "Vector.cuh"
#include "Matrix.cuh"

namespace Cudheart::VectorOps {

	template <typename T>
	Vector<T>* asVector(T* arr, int len) {
		return new Vector<T>(arr, len);
	}

	template <typename T>
	Vector<T>* empty(int len) {
		return new Vector<T>(len);
	}

	template <typename T>
	Vector<T>* emptyLike(Vector<T>* arr) {
		return empty<T>(arr->getSize());
	}

	// do not fucking use this with T = char, string or custom class
	// if you do, prepare for trouble and make it double
	template <typename T>
	Vector<T>* arange(T start, T end, T jump) {
		int len = ((end - start) / jump);
		Vector<T>* out = empty<T>(len);

		for (int i = 0; start < end; start += jump) {
			out->set(i++, start);
		}

		return out;
	}

	template <typename T>
	Vector<T>* arange(T end, T jump) {
		return arange<T>((T)0, end, jump);
	}

	template <typename T>
	Vector<T>* arange(T end) {
		return arange<T>((T)0, end, (T)1);
	}

	template <typename T>
	Vector<T>* full(int s, T value) {
		Vector<T>* out = empty<T>(s);

		for (int i = 0; i < s; i++) {
			out->set(i, value);
		}

		return out;
	}

	template <typename T>
	Vector<T>* fullLike(Vector<T>* arr, T value) {
		return full<T>(arr->getSize(), value);
	}

	template <typename T>
	Vector<T>* fromMatrix(Matrix<T>* mat) {
		Vector<T>* out = zeros(mat->getSize());

		for (int i = 0; i < mat->getSize(); i++) {
			out->set(i, mat->get(i));
		}

		return out;
	}

	template <typename T>
	Vector<T>* linspace(T start, T stop, T num, bool endpoint) {
		T jump = (stop - start) / num;
		if (endpoint) {
			jump = (stop - start) / (num - 1);
			stop += jump;
		}
		return arange<T>(start, stop, jump);
	}

	template <typename T>
	Vector<T>* linspace(T start, T stop, T num) {
		return linspace<T>(start, stop, num, true);
	}

	template <typename T>
	Vector<T>* linspace(T start, T stop) {
		return linspace<T>(start, stop, (T)50, true);
	}

	template <typename T>
	Vector<T>* ones(int shape) {
		return full<T>(shape, 1);
	}

	template <typename T>
	Vector<T>* onesLike(Vector<T>* arr) {
		return ones<T>(arr->getSize());
	}

	template <typename T>
	Vector<T>* zeros(int len) {
		return full<T>(len, 0);
	}

	template <typename T>
	Vector<T>* zerosLike(Vector<T>* arr) {
		return zeros<T>(arr->getSize());
	}
};