#pragma once

#include "Vector.h"
#include "../Dtypes/Dtypes.h"

class ArrayOps {
	friend class Dtype;
	friend class Vector;
public:
	static Vector asarray(int* arr, int size, Dtype *dtype, bool copy);

	static Vector asarray(int* arr, int size, bool copy) {
		return asarray(arr, size, new DInt32(), copy);
	}

	static Vector asarray(int* arr, int size, Dtype *dtype) {
		return asarray(arr, size, dtype, false);
	}

	static Vector asarray(int* arr, int size) {
		return asarray(arr, size, new DInt32(), false);
	}

	static Vector arange(double low, double high, double jump);

	static Vector arange(double low, double high) {
		return arange(low, high, 1);
	}

	static Vector arange(double high) {
		return arange(0, high, 1);
	}
};
