#pragma once

#include "Vector.h"
#include "VectorInt.h"
#include "../Dtypes/Dtypes.h"

class ArrayOps {
	friend class Dtype;
	friend class Vector;
	friend class VectorInt;
public:
	static VectorInt asarray(int* arr, int size, Dtype dtype, bool copy);

	static VectorInt asarray(int* arr, int size, bool copy) {
		return asarray(arr, size, Dtype::determine(arr), copy);
	}

	static VectorInt asarray(int* arr, int size, Dtype dtype) {
		return asarray(arr, size, dtype, false);
	}

	static VectorInt asarray(int* arr, int size) {
		return asarray(arr, size, Dtype::determine(arr), false);
	}

	static Vector arange(double low, double high, double jump);
};
