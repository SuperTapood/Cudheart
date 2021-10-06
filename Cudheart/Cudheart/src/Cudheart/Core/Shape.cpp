#include "Shape.h"

Shape::Shape(int* shape)
{
	this->shape = shape;

	__int64 len = *(&shape + 1) - shape;
	__int64 l = 0;

	for (__int64 i = 0; i < len; i++) {
		if (shape[i] == 0) {
			length = l;
			// bc what the fck does a shape of (5, 0, 5) even mean
			break;
		}
	}

	length = len;
}

Shape::~Shape()
{
	delete[] shape;
}

const int Shape::at(int idx)
{
	return shape[idx];
}
