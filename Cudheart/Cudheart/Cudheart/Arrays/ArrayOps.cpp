#include "ArrayOps.h"
#include "../Inc.h"


Vector ArrayOps::asarray(void* arr, int size, Dtype *dtype, bool copy)
{
	if (copy) {
		arr = dtype->copy(arr, size);
	}
	return Vector((void*)arr, size);
}

Vector ArrayOps::arange(double low, double high, double jump, Dtype *dtype)
{
	int size = (high - low) / jump;
	static Vector v = empty(size, dtype);
	v.setCopied(true);

	for (int i = 0; i < v.size; i++) {
		v.set(i, &low);
		low += jump;
	}

	return v;
}

Vector ArrayOps::empty(int size, Dtype* dtype)
{
	return asarray((void*)dtype->empty(size), size);
}
