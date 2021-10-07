#include "Shape.h"
#include "../Utils/exception.h"

using namespace Cudheart::Utils::Exceptions;

Shape::Shape(int shape[])
{
	for (int i = 0; i < 4; i++) {
		this->shape[i] = shape[i];
	}

	__int64 len = *(&shape + 1) - shape;
	__int64 l = 0;

	for (__int64 i = 0; i < len; i++) {
		if (shape[i] == 0) {
			BadShape();
			length = l;
			// bc what the fck does a shape of (5, 0, 5) even mean
			break;
		}
	}

	length = len;


	for (int i = 0; i < len; i++) {
		size += shape[i];
	}
}

Shape::~Shape()
{
	delete[] shape;
}

const int Shape::at(int idx)
{
	return shape[idx];
}
