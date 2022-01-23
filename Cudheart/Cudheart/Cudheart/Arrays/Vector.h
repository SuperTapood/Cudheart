#pragma once

#include "../Inc.h"
#include "../Dtypes/Dtype.h"

class Vector {
public:
	int size;
	Dtype *dtype;
//private:
	void* arr;
public:
	Vector(int size);

	Vector(void* arr, int size);

	Vector(void* arr, int size, Dtype *dtype);

	~Vector();

	string toString();

	friend ostream& operator<<(ostream& out, const Vector &v);

	void* operator[](size_t i);

	void set(size_t i, void* value);
};