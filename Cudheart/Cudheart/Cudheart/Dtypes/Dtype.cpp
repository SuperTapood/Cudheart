#include "Dtype.h"

#include "Dint.h"
#include "../Arrays/VectorInt.h"

class Dint;

Dtype Dtype::determine(int* arr) {
	return Dint();
}

VectorInt Dtype::asVector(int* arr, int size)
{
	return VectorInt(arr, size);
}
