#pragma once

#include "../Util.cuh"
#include "Array.cuh"

template <typename T>
class ArrayOps {
public:

	static Array<T>* asarray(T* arr, Shape* shape) {
		return new Array<T>(arr, shape->clone());
	}

	// do not fucking use this with T = char, string or custom class
	// if you do, prepare for trouble and make it double
	static Array<T>* arange(T start, T end, T jump) {
		int len = ((end - start) / jump);
		Array<T>* out = empty(new Shape(new int[] {len}, 1));

		for (int i = 0; start < end; start += jump) {
			out->setAbs(i++, start);
		}

		return out;
	}

	static Array<T>* arange(T end, T jump) {
		return arange((T)0, end, jump);
	}

	static Array<T>* arange(T end) {
		return arange((T)0, end, (T)1);
	}

	static Array<T>* empty(Shape* shape) {
		return new Array<T>(shape);
	}

	static Array<T>* emptyLike(Array<T>* arr) {
		return empty(arr->cloneShape());
	}

	static Array<T>* eye(int rows, int cols, int k)
	{
		Array<T>* out = zeros(new Shape(new int[] {rows, cols}, 2));

		for (int i = 0; k < cols && i < rows; i++, k++) {
			out->setAbs(i * rows + k, 1);
		}

		return out;
	}


	static Array<T>* eye(int rows) {
		return eye(rows, rows, 0);
	}


	static Array<T>* eye(int rows, int k) {
		return eye(rows, rows, k);
	}

	static Array<T>* full(Shape* shape, T value) {
		Array<T>* out = empty(new Shape(new int[] {shape->getSize()}, 1));

		for (int i = 0; i < shape->getSize(); i++) {
			out->setAbs(i, value);
		}

		return (*out).reshape(shape);
	}

	static Array<T>* fullLike(Array<T>* arr, T value) {
		return full(arr->shape->clone(), value);
	}

	static Array<T>* linspace(T start, T stop, T num, bool endpoint) {
		T jump = (stop - start) / num;
		if (endpoint) {
			jump = (stop - start) / (num - 1);
			stop += jump;
		}
		return arange(start, stop, jump);
	}

	static Array<T>* linspace(T start, T stop, T num) {
		return linspace(start, stop, num, true);
	}

	static Array<T>* linspace(T start, T stop) {
		return linspace(start, stop, (T)50, true);
	}

	static Array<T>* meshgrid(Array<T>* a, Array<T>* b)
	{
		if ((*a).getDims() != 1 || (*b).getDims() != 1) {
			return nullptr;
		}
		int alen = a->getShape(0);
		int blen = b->getShape(0);
		Array<T>* out = (Array<T>*)malloc(sizeof(Array<T>) * 2);
		out[0] = *zeros(new Shape(new int[] {blen, alen}, 2));
		out[1] = *zeros(new Shape(new int[] {blen, alen}, 2));

		for (int i = 0; i < blen; i++) {
			
		}
		return out;
	}

	static Array<T>* ones(Shape* shape) {
		return full(shape, 1);
	}

	static Array<T>* onesLike(Array<T>* arr) {
		return ones(arr->shape->clone());
	}

	static Array<T>* zeros(Shape* shape) {
		return full(shape, 0);
	}

	static Array<T>* zerosLike(Array<T>* arr) {
		return zeros(arr->shape->clone());
	}
};