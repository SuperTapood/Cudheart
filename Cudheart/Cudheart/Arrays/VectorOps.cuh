#pragma once

#include "../Util.cuh"
#include "Vector.cuh"
#include "Matrix.cuh"

template <typename T>
class VectorOps {
public:

	static Vector<T>* asVector(T* arr, int len) {
		return new Vector<T>(arr, len);
	}

	// do not fucking use this with T = char, string or custom class
	// if you do, prepare for trouble and make it double
	static Vector<T>* arange(T start, T end, T jump) {
		int len = ((end - start) / jump);
		Vector<T>* out = empty(len);

		for (int i = 0; start < end; start += jump) {
			out->set(i++, start);
		}

		return out;
	}

	static Vector<T>* arange(T end, T jump) {
		return arange((T)0, end, jump);
	}

	static Vector<T>* arange(T end) {
		return arange((T)0, end, (T)1);
	}

	static Vector<T>* empty(int len) {
		return new Vector<T>(len);
	}

	static Vector<T>* emptyLike(Vector<T>* arr) {
		return empty(arr->getSize());
	}

	static Vector<T>* full(int s, T value) {
		Vector<T>* out = empty(s);

		for (int i = 0; i < s; i++) {
			out->set(i, value);
		}

		return out;
	}

	static Vector<T>* fullLike(Vector<T>* arr, T value) {
		return full(arr->getSize(), value);
	}

	static Vector<T>* fromMatrix(Matrix<T>* mat) {
		Vector<T>* out = zeros(mat->getSize());

		for (int i = 0; i < mat->getSize(); i++) {
			out->set(i, mat->get(i));
		}

		return out;
	}

	static Vector<T>* linspace(T start, T stop, T num, bool endpoint) {
		T jump = (stop - start) / num;
		if (endpoint) {
			jump = (stop - start) / (num - 1);
			stop += jump;
		}
		return arange(start, stop, jump);
	}

	static Vector<T>* linspace(T start, T stop, T num) {
		return linspace(start, stop, num, true);
	}

	static Vector<T>* linspace(T start, T stop) {
		return linspace(start, stop, (T)50, true);
	}

	static Vector<T>* ones(int shape) {
		return full(shape, 1);
	}

	static Vector<T>* onesLike(Vector<T>* arr) {
		return ones(arr->getSize());
	}

	static Vector<T>* zeros(int len) {
		return full(len, 0);
	}

	static Vector<T>* zerosLike(Vector<T>* arr) {
		return zeros(arr->getSize());
	}
};