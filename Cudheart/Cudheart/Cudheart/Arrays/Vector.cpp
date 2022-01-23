#include "Vector.h"
#include "../Dtypes/Dtypes.h"

Vector::Vector(int size) {
	this->arr = nullptr;
	this->size = size;
	this->dtype = new DInt32();
}

Vector::Vector(void* arr, int size) {
	this->arr = arr;
	this->size = size;
	this->dtype = new DInt32();
}

Vector::Vector(void* arr, int size, Dtype *dtype) {
	this->arr = arr;
	this->size = size;
	this->dtype = dtype;
}

Vector::~Vector()
{
	delete dtype;
}

void* Vector::operator[](size_t i)
{
	return (*dtype).get(arr, i);
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
