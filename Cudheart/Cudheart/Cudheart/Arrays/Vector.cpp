#include "Vector.h"
#include "../Dtypes/Dtypes.h"
#include "../Exceptions/Exceptions.h"

Vector::Vector(int size) {
	this->arr = nullptr;
	this->size = size;
	this->dtype = new DInt();
}

Vector::Vector(void* arr, int size) {
	this->arr = arr;
	this->size = size;
	this->dtype = new DInt();
}

Vector::Vector(void* arr, int size, Dtype *dtype) {
	this->arr = arr;
	this->size = size;
	this->dtype = dtype;
}

Vector::~Vector()
{
	if (!copied) {
		delete dtype;
		delete arr;
	}
}

void* Vector::operator[](size_t i)
{
	if (i < 0) {
		return this->operator[](size + i);
	}
	if (i > size) {
		throw IndexError(i, 0, size);
	}
	return (*dtype).get(arr, i);
}

void Vector::set(size_t i, void* value)
{
	dtype->set(arr, i, value);
}


string Vector::toString() {
	ostringstream os;
	os << this;
	return os.str();
}

ostream& operator<<(ostream& out, const Vector& vec)
{
	out << vec.dtype->getName() << " Vector[";
	for (unsigned int j = 0; j < vec.size; j++)
	{
		if (j % vec.size == vec.size - 1)
			out << (*vec.dtype).toString(vec.arr, j) << "]" << endl;
		else
			out << (*vec.dtype).toString(vec.arr, j) << ", ";
	}
	return out;
}
