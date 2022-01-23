#pragma once

#include "Vector.h"
#include "../Dtypes/Dtypes.h"

class ArrayOps {
	friend class Dtype;
	friend class Vector;
public:
	static Vector asarray(void* arr, int size, Dtype *dtype, bool copy);

	static Vector asarray(void* arr, int size, bool copy) {
		return asarray(arr, size, new DInt(), copy);
	}

	static Vector asarray(void* arr, int size, Dtype *dtype) {
		return asarray(arr, size, dtype, false);
	}

	static Vector asarray(void* arr, int size) {
		return asarray(arr, size, new DInt(), false);
	}

	static Vector arange(double low, double high, double jump, Dtype *dtype);

	static Vector arange(double low, double high, double jump) {
		return arange(low, high, jump, new DDouble());
	}

	static Vector arange(double low, double high) {
		return arange(low, high, 1);
	}

	static Vector arange(double high) {
		return arange(0, high, 1);
	}

	static Vector empty(int size, Dtype *dtype);

	static Vector empty(int size) {
		return empty(size, new DInt());
	}
};
