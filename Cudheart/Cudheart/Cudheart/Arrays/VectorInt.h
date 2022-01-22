#pragma once

#include "Vector.h"


class VectorInt : public Vector {
private:
	int* arr;
public:
	VectorInt(int* arr, int size);
	VectorInt(int* arr);
	/*~VectorInt() {
		delete arr;
	}*/
	int operator[](size_t i);
	void set(size_t i, int value);
	string toString();
};