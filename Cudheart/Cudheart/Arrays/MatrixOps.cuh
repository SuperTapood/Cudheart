#pragma once

#include "../Util.cuh"
#include "Matrix.cuh"
#include "Vector.cuh"
#include "VectorOps.cuh"


// all this class does is call VectorOps functions and cast the resulting vector to a matrix lol

template <typename T>
class MatrixOps {
public:

	static Matrix<T>* asMatrix(T* arr, int width, int height) {
		return new Matrix<T>(arr, width, height);
	}

	// do not fucking use this with T = char, string or custom class
	// if you do, prepare for trouble and make it double
	static Matrix<T>* arange(T start, T end, T jump, int width, int height) {
		return asMatrix(VectorOps<T>::arange(start, end, jump), width, height);
	}

	static Matrix<T>* arange(T end, T jump, int width, int height) {
		return arange((T)0, end, jump, width, height);
	}

	static Matrix<T>* arange(T end, int width, int height) {
		return arange((T)0, end, (T)1, width, height);
	}

	static Matrix<T>* empty(int width, int height) {
		return new Matrix<T>(width, height);
	}

	static Matrix<T>* emptyLike(Matrix<T>* mat) {
		return empty(mat->getWidth(), mat->getHeight());
	}

	static Matrix<T>* full(int width, int height, T value) {
		return asMatrix(VectorOps<T>::full(width * height, value), width, height);
	}

	static Matrix<T>* fullLike(Matrix<T>* mat, T value) {
		return full(mat->getWidth(), mat->getHeight(), value);
	}

	static Matrix<T>* fromVector(Vector<T>* vec, int width, int height) {
		if (width * height != vec->getSize()) {
			cout << "fromVector failed with w: " << width << " and h: " << height << "with size: " << vec->getSize() << endl;
			return nullptr;
		}
		Matrix<T>* out = empty(width, height);

		for (int i = 0; i < out->getSize(); i++) {
			out->set(i, vec->get(i));
		}

		return out;
	}

	static Matrix<T>* linspace(T start, T stop, T num, bool endpoint, int width, int height) {
		return asMatrix(VectorOps<T>::linspace(start, stop, num, endpoint), width, height);
	}

	static Matrix<T>* linspace(T start, T stop, T num) {
		return linspace(start, stop, num, true);
	}

	static Matrix<T>* linspace(T start, T stop) {
		return linspace(start, stop, (T)50, true);
	}

	static Matrix<T>* ones(int width, int height) {
		return full(width, height, 1);
	}

	static Matrix<T>* onesLike(Matrix<T>* mat) {
		return ones(mat->getWidth(), mat->getHeight());
	}

	static Matrix<T>* zeros(int width, int height) {
		return full(width, height, 0);
	}

	static Matrix<T>* zerosLike(Matrix<T>* mat) {
		return zeros(mat->getWidth(), mat->getHeight());
	}
};