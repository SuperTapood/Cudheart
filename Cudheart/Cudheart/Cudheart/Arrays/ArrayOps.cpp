#include "ArrayOps.h"
#include "../Inc.h"


Array ArrayOps::asarray(void* arr, Shape *shape, Dtype *dtype, bool copy)
{
	if (copy) {
		arr = dtype->copy(arr, shape);
	}
	return Array((void*)arr, shape, dtype);
}

Array ArrayOps::arange(double low, double high, double jump, Dtype *dtype)
{
	int size = (high - low) / jump;
	static Array v = empty(new Shape({size}), dtype);
	v.setCopied(true);

	for (int i = 0; i < v.size; i++) {
		v.set(i, &low);
		low += jump;
	}

	return v;
}

Array ArrayOps::empty(Shape* shape, Dtype* dtype)
{
	return asarray((void*)dtype->empty(shape), shape, dtype);
}
