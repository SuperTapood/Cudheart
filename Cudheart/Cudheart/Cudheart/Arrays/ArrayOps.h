#pragma once

#include "Array.h"
#include "../Exceptions/NotImplementedError.h"


template <typename T>
class ArrayOps {
	friend class Array<T>;
public:
	static Array<T>* asarray(T* arr, Shape* shape)
	{
		return new Array<T>(arr, shape);
	}

	static Array<T>* arange(T low, T high, T jump)
	{
		int size = (high - low) / jump;
		cout << "AAAAAAAA" << endl;
		Array<T>* out = empty(new Shape({ size }));
		(*out).setCopied(true);
		// god knows why i need a temp variable but yeah
		T t = low;
		for (int i = 0; i < (*out).size; i++) {
			(*out).setAbsolute(i, t);
			t += jump;
		}
		cout << size << endl;
		cout << (*out).size << endl;
		cout << (*out).getShapeString() << endl;

		return out;
	}

	static Array<T>* arange(T low, T high) {
		return arange(low, high, 1);
	}

	
	static Array<T>* arange(T high) {
		return arange(0, high, 1);
	}

	
	static Array<T>* empty(Shape* shape) {
		T* arr = (T*)malloc(shape->size * sizeof(T));\
		Array<T>* out = new Array<T>(arr, shape);
		return out;
	}

	
	static Array<T>* emptyLike(Array<T>* arr) {
		return empty(arr->dupeShape());
	}

	
	static Array<T>* eye(int rows, int cols, int k)
	{
		Array<T>* out = zeros(new Shape(rows, cols));

		int one = 1;
		for (int i = 0; k < cols && i < rows; k++, i++) {
			k = k % cols;
			(*out).set(&one, 2, i, k);
		}

		return out;
	}

	
	static Array<T>* eye(int rows) {
		return eye(rows, rows, 0);
	}

	
	static Array<T>* eye(int rows, int k) {
		return eye(rows, rows, k);
	}

	
	static Array<T>* full(Shape* shape, T value)
	{
		Array<T>* out = empty(shape);
		for (int i = 0; i < (*out).size; i++) {
			(*out).setAbsolute(i, value);
		}
		return out;
	}

	
	static Array<T>* fullLike(Array<T>* arr, T value) {
		return full(arr->dupeShape(), value);
	}

	
	static Array<T>* linspace(T min, T max, int steps) {
		return arange(min, max, (max - min) / steps);
	}

	
	static Array<T>* meshgrid(Array<T>* a, Array<T>* b)
	{
		if ((*a).getDims() != 1 || (*b).getDims() != 1) {
			return nullptr;
		}
		int alen = (*a).getShapeAt(0);
		int blen = (*b).getShapeAt(0);
		static Array<T>* out = (Array<T>*)malloc(sizeof(Array<T>) * 2);
		out[0] = *zeros(new Shape(blen, alen));
		out[1] = *zeros(new Shape(blen, alen));

		for (int i = 0; i < blen; i++) {
			for (int j = 0; j < alen; j++) {
				out[0].set((*a).getFlat(j), 2, i, j);
			}
		}

		for (int i = 0; i < alen; i++) {
			for (int j = 0; j < blen; j++) {
				out[1].set((*b).getFlat(j), 2, j, i);
			}
		}

		return out;
	}

	
	static Array<T>* ones(Shape* shape) {
		int ones = 1;
		return full(shape, ones);
	}

	
	static Array<T>* onesLike(Array<T>* arr) {
		return ones(arr->dupeShape());
	}

	
	static Array<T>* tril(Array<T>* arr, int k)
	{
		if ((*arr).getDims() != 2) {
			return nullptr;
		}
		int zero = 0;
		Array<T> out = *(*arr).dupe();
		for (int i = 0; i < out.getShapeAt(0); i++) {
			for (int j = i + 1 + k; j < out.getShapeAt(1); j++) {
				cout << "i: " << i << " j: " << j << endl;
				cout << "shape: " << out.getShapeString() << endl;
				out.set(zero, 2, i, j);
			}
		}
		return &out;
	}

	
	static Array<T>* tril(Array<T>* arr) {
		return tril(arr, 0);
	}

	
	static Array<T>* triu(Array<T>* arr, int k)
	{
		Array<T> out = *(*arr).dupe();
		int zero = 0;
		for (int i = 0; i < out.getShapeAt(0); i++) {
			for (int j = i + 1 + k; j < out.getShapeAt(1); j++) {
				out.set(zero, 2, i, j);
			}
		}
		return &out;
	}
	
	
	static Array<T>* triu(Array<T>* arr) {
		return triu(arr, 0);
	}

	
	static Array<T>* zeros(Shape* shape) {
		int v = 0;
		return full(shape, v);
	}

	
	static Array<T>* zerosLike(Array<T>* arr) {
		return zeros(arr->dupeShape());
	}
};