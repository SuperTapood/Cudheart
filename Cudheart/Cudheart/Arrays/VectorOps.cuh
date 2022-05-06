#pragma once

#include "../Util.cuh"
#include "Vector.cuh"
#include "Matrix.cuh"
#include "../Exceptions/Exceptions.cuh"

namespace Cudheart::VectorOps {
	using NDArrays::Vector;
	using NDArrays::Matrix;

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
		int len = (int)((end - start) / jump);
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
			out->set(i, mat->getAbs(i));
		}

		return out;
	}

	template <typename T>
	Vector<T>* fromMatrix(Matrix<T>* mat, bool destroy) {
		Vector<T>* res = fromMatrix(mat);

		if (destroy) {
			delete mat;
		}

		return res;
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
	Vector<T>* ones(int len) {
		return full<T>(len, 1);
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

	template <typename T>
	Vector<T>* logspace(T start, T stop, T num, bool endpoint, double base) {
		T jump = (stop - start) / num;
		if (endpoint) {
			jump = (stop - start) / (num - 1);
			stop += jump;
		}
		int len = (int)((end - start) / jump);
		Vector<T>* out = empty<T>(len);

		for (int i = 0; start < end; start += jump) {
			out->set(i++, pow(base, start));
		}

		return out;
	}

	template <typename T>
	Vector<T>* logspace(T start, T stop, T num, bool endpoint) {
		return logspace<T>(start, stop, num, endpoint, 10.0);
	}

	template <typename T>
	Vector<T>* logspace(T start, T stop, T num) {
		return logspace<T>(start, stop, num, true, 10.0);
	}

	template <typename T>
	Vector<T>* logspace(T start, T stop, T num, double base) {
		return logspace<T>(start, stop, num, true, base);
	}

	template <typename T>
	Vector<T>* logspace(T start, T stop, double base) {
		return logspace<T>(start, stop, (T)50, true, base);
	}

	template <typename T>
	Vector<T>* logspace(T start, T stop) {
		return logspace<T>(start, stop, (T)50, true, 10.0);
	}

	template <typename T>
	Vector<T>* geomspace(T start, T stop, T num, bool endpoint) {
		start = log10(start);
		stop = log10(stop);
		auto res = logspace(start, stop, num, endpoint);
		res.set(0, start);
		if (endpoint) {
			res.set(-1, stop);
		}
		return res;
	}

	template <typename T>
	Vector<T>* geomspace(T start, T stop, T num) {
		return geomspace<T>(start, stop, num, true);
	}

	template <typename T>
	Vector<T>* geomspace(T start, T stop) {
		return geomspace<T>(start, stop, (T)50, true);
	}
};