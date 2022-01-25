#pragma once

#include "Array.h"
#include "../Dtypes/Dtypes.h"
#include "../Exceptions/NotImplementedError.h"

class ArrayOps {
	friend class Dtype;
	friend class Array;
public:
	static Array asarray(void* arr, Shape* shape, Dtype *dtype);

	static Array asarray(void* arr, Shape* shape) {
		return asarray(arr, shape, new DInt());
	}

	static Array arange(double low, double high, double jump, Dtype *dtype);

	static Array arange(double low, double high, double jump) {
		return arange(low, high, jump, new DDouble());
	}

	static Array arange(double low, double high) {
		return arange(low, high, 1);
	}

	static Array arange(double high) {
		return arange(0, high, 1);
	}

	static Array empty(Shape* shape, Dtype *dtype);

	static Array empty(Shape* shape) {
		return empty(shape, new DInt());
	}

	static Array emptyLike(Array* arr, Dtype* dtype) {
		return empty(arr->shape, dtype);
	}

	static Array emptyLike(Array* arr) {
		return emptyLike(arr, arr->dtype->dupe());
	}

	static Array eye(int rows, int cols, int k, Dtype* dtype);

	static Array eye(int rows, int k, Dtype* dtype) {
		return eye(rows, rows, k, dtype);
	}

	static Array eye(int rows, Dtype* dtype) {
		return eye(rows, rows, 0, dtype);
	}

	static Array eye(int rows, int cols, int k) {
		return eye(rows, cols, k, new DInt());
	}

	static Array eye(int rows, int k) {
		return eye(rows, rows, k, new DInt());
	}

	static Array eye(int rows) {
		return eye(rows, rows, 0, new DInt());
	}

	static Array full(Shape* shape, void* value, Dtype* dtype);

	static Array full(Shape* shape, void* value) {
		return full(shape, value, new DInt());
	}

	static Array fullLike(Array* arr, void* value, Dtype* dtype) {
		return full(arr->shape, value, dtype);
	}

	static Array fullLike(Array* arr, void* value) {
		return fullLike(arr, value, arr->dtype->dupe());
	}

	static Array linspace(double min, double max, int steps, Dtype* dtype) {
		return arange(min, max, (max - min) / steps, dtype);
	}

	static Array linspace(double min, double max, int steps) {
		return linspace(min, max, steps, new DInt());
	}

	static Array meshgrid(Array a, Array b) {
		// needs array to actually support matrix stuff
		// assert a and b are vectors we are gonna do this very simply
		// https://stackoverflow.com/questions/36013063/what-is-the-purpose-of-meshgrid-in-python-numpy
		// ^ for guidance
		throw NotImplementedError("meshgrid");
	}

	static Array ones(Shape* shape, Dtype* dtype) {
		return full(shape, (void*)1, dtype);
	}

	static Array ones(Shape* shape) {
		return ones(shape, new DInt());
	}

	static Array onesLike(Array* arr, Dtype* dtype) {
		return ones(arr->shape, dtype);
	}

	static Array onesLike(Array* arr) {
		return onesLike(arr, arr->dtype->dupe());
	}

	static Array tril(Array* arr, int k) {
		// need ndim perception
		// todo!
		throw NotImplementedError("eye");
	}

	static Array tril(Array* arr) {
		return tril(arr, 0);
	}

	static Array zeros(Shape* shape, Dtype* dtype) {
		int a = 5;
		void* p;
		memcpy(&p, &a, sizeof a);
		return full(shape, p, dtype);
	}

	static Array zeros(Shape* shape) {
		return zeros(shape, new DInt());
	}

	static Array zerosLike(Array* arr, Dtype* dtype) {
		return zeros(arr->shape, dtype);
	}

	static Array zerosLike(Array* arr) {
		return zerosLike(arr, arr->dtype->dupe());
	}
};
