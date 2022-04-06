#pragma once

#include "../Util.cuh"
#include "Matrix.cuh"
#include "Vector.cuh"
#include "VectorOps.cuh"
#include "../Exceptions/Exceptions.cuh"


// all this class does is call VectorOps functions and cast the resulting vector to a matrix lol

namespace Cudheart::MatrixOps {
	using NDArrays::Matrix;
	using NDArrays::Vector;

	template <typename T>
	Matrix<T>* asMatrix(T* arr, int width, int height) {
		return new Matrix<T>(arr, width, height);
	}

	// do not fucking use this with T = char, string or custom class
	// if you do, prepare for trouble and make it double
	template <typename T>
	Matrix<T>* arange(T start, T end, T jump, int width, int height) {
		return asMatrix(VectorOps::arange<T>(start, end, jump), width, height);
	}

	template <typename T>
	Matrix<T>* arange(T end, T jump, int width, int height) {
		return arange((T)0, end, jump, width, height);
	}

	template <typename T>
	Matrix<T>* arange(T end, int width, int height) {
		return arange((T)0, end, (T)1, width, height);
	}

	template <typename T>
	Matrix<T>* empty(int width, int height) {
		return new Matrix<T>(width, height);
	}

	template <typename T>
	Matrix<T>* emptyLike(Matrix<T>* mat) {
		return empty(mat->getWidth(), mat->getHeight());
	}

	template <typename T>
	Matrix<T>* full(int width, int height, T value) {
		return asMatrix(VectorOps::full<T>(width * height, value), width, height);
	}

	template <typename T>
	Matrix<T>* fullLike(Matrix<T>* mat, T value) {
		return full(mat->getWidth(), mat->getHeight(), value);
	}

	template <typename T>
	Matrix<T>* fromVector(Vector<T>* vec, int width, int height) {
		if (width * height != vec->getSize()) {
			throw Exceptions::MatrixConversionException(width, height, vec->getSize());
		}
		Matrix<T>* out = empty<T>(width, height);

		for (int i = 0; i < out->getSize(); i++) {
			out->set(i, vec->get(i));
		}

		return out;
	}

	template <typename T>
	Matrix<T>* fromVector(Vector<T>* vec, int width, int height, bool destroy) {
		Matrix<T>* res = fromVector(vec, width, height);
		if (destroy) {
			delete vec;
		}
		return res;
	}

	template <typename T>
	Matrix<T>* linspace(T start, T stop, T num, bool endpoint, int width, int height) {
		return asMatrix(VectorOps::linspace<T>(start, stop, num, endpoint), width, height);
	}

	template <typename T>
	Matrix<T>* linspace(T start, T stop, T num) {
		return linspace(start, stop, num, true);
	}

	template <typename T>
	Matrix<T>* linspace(T start, T stop) {
		return linspace(start, stop, (T)50, true);
	}

	template <typename T>
	Matrix<T>* ones(int width, int height) {
		return full(width, height, 1);
	}

	template <typename T>
	Matrix<T>* onesLike(Matrix<T>* mat) {
		return ones(mat->getWidth(), mat->getHeight());
	}

	template <typename T>
	Matrix<T>* zeros(int width, int height) {
		return full(width, height, 0);
	}

	template <typename T>
	Matrix<T>* zerosLike(Matrix<T>* mat) {
		return zeros(mat->getWidth(), mat->getHeight());
	}

	template <typename T>
	Matrix<T>* logspace(T start, T stop, T num, bool endpoint, double base, int width, int height) {
		return fromVector(VectorOps::logspace<T>(start, stop, num, endpoint, base), width, height);
	}

	template <typename T>
	Matrix<T>* logspace(T start, T stop, T num, bool endpoint, int width, int height) {
		return logspace<T>(start, stop, num, endpoint, 10.0, width, height);
	}

	template <typename T>
	Matrix<T>* logspace(T start, T stop, T num, int width, int height) {
		return logspace<T>(start, stop, num, true, 10.0, width, height);
	}

	template <typename T>
	Matrix<T>* logspace(T start, T stop, T num, double base, int width, int height) {
		return logspace<T>(start, stop, num, true, base, width, height);
	}

	template <typename T>
	Matrix<T>* logspace(T start, T stop, double base, int width, int height) {
		return logspace<T>(start, stop, (T)50, true, base, width, height);
	}

	template <typename T>
	Matrix<T>* logspace(T start, T stop, int width, int height) {
		return logspace<T>(start, stop, (T)50, true, 10.0, width, height);
	}
};