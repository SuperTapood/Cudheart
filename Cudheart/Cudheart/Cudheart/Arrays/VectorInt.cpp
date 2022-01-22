#include "VectorInt.h"

VectorInt::VectorInt(int* arr, int size) : Vector(size)
{
	this->arr = arr;
}

int VectorInt::operator[](size_t i)
{
	return arr[i];
}

void VectorInt::set(size_t i, int value)
{
	arr[i] = value;
}

string VectorInt::toString() {
	VectorInt vec = *this;
	ostringstream out;
	out << "VectorInt[";
	for (unsigned int j = 0; j < vec.size; j++)
	{
		if (j % vec.size == vec.size - 1)
			out << vec[j] << "]";
		else
			out << vec[j] << ", ";
	}
	return out.str();
}
