#pragma once

#include "Array.h"
#include "../Dtypes/Dtypes.h"

class ArrayOps {
	friend class Dtype;
	friend class Array;
public:
	static Array asarray(void* arr, Shape* shape, Dtype *dtype, bool copy);

	static Array asarray(void* arr, Shape* shape, bool copy) {
		return asarray(arr, shape, new DInt(), copy);
	}

	static Array asarray(void* arr, Shape* shape, Dtype *dtype) {
		return asarray(arr, shape, dtype, false);
	}

	static Array asarray(void* arr, Shape* shape) {
		return asarray(arr, shape, new DInt(), false);
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
};
