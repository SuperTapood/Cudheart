#pragma once

#include "../Arrays/Vector.h"

class Dtype {
	friend class VectorInt;
public:
public:
	static Dtype determine(int* arr);
	static VectorInt asVector(int* arr, int size);
};
