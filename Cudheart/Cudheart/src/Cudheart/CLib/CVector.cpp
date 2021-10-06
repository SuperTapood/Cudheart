

#include "CVector.h"
#include <stdio.h>

#include "../Core/Shape.h"
#include "../Dtypes/Dtype.h"


CVector::CVector(void* arr, Shape shape, Dtype dtype, bool copy) {
	init(arr, shape, dtype, copy);
}

CVector::CVector(void* arr, Shape shape, Dtype dtype)
{
	init(arr, shape, dtype, true);
}

CVector::~CVector()
{
	delete[] this->arr;
}

void CVector::init(void* arr, Shape shape, Dtype dtype, bool copy)
{
	if (shape.length > 0) {
		printf("shape with more than 1 dimension given. using the first dimension.");
	}
	this->arr = dtype.cast(arr, shape.at(0), copy);
	this->shape = shape;
	this->dtype = dtype;
}
