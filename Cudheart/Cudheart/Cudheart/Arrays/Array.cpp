#include "Array.h"
#include "../Dtypes/Dtypes.h"
#include "../Exceptions/Exceptions.h"

Array::Array(Shape* shape) {
	this->arr = malloc(size * sizeof(int));
	this->size = (*shape).size;
	this->shape = shape;
	this->dtype = new DInt();
}

Array::Array(Shape* shape, Dtype *dtype) {
	this->arr = malloc(size * (*dtype).getSize());
	this->size = (*shape).size;
	this->shape = shape;
	this->dtype = dtype;
}

Array::Array(void* arr, Shape* shape) {
	this->dtype = new DInt();
	this->arr = dtype->copy(arr, shape);
	this->size = (*shape).size;
	this->shape = shape;
}

Array::Array(void* arr, Shape* shape, Dtype *dtype) {
	this->arr = dtype->copy(arr, shape);
	this->size = (*shape).size;
	this->shape = shape->dupe();
	this->dtype = dtype->dupe();
}

Array::~Array()
{
	if (!true) {
		delete arr;
		delete dtype;
		delete shape;
	}
}

void* Array::operator[](size_t i)
{
	if (i < 0) {
		return this->operator[](size + i);
	}
	if (i > size) {
		throw IndexError(i, 0, size);
	}
	return (*dtype).get(arr, i);
}

void Array::setAbsolute(size_t i, void* value)
{
	cout << i << endl;
	cout << shape->toString() << endl;
	dtype->set(arr, i, value);
}

string Array::asString(size_t i)
{
	return dtype->toString(arr, i);
}

string Array::getShapeString()
{
	return shape->toString();
}

void Array::reshape(Shape* shape)
{
	if ((*shape).size != this->shape->size) {
		throw ShapeError(this->shape, shape);
	}

	this->shape = shape;
}

//void* Array::get(size_t i)
//{
//	return operator[](i);
//}

void* Array::get(int len, ...)
{
	// for now
	if (len != shape->length) {
		throw IndexError(len, shape);
	}
	int index = 0;
	va_list args;
	va_start(args, len);
	for (int i = 0; len > i; i++) {
		index += va_arg(args, int) * ((*shape).sizeFrom(i));
	}

	return operator[](index);
}

void Array::setAbsolute(void* value, int len, ...)
{
	// for now
	if (len != shape->length) {
		throw IndexError(len, shape);
	}
	int index = 0;
	va_list args;
	va_start(args, len);
	for (int i = 0; len > i; i++) {
		index += va_arg(args, int) * ((*shape).sizeFrom(i));
	}

	dtype->set(arr, index, value);
}

bool Array::operator==(Array &v)
{
	if (v.size != size || v.dtype->getName() != dtype->getName()) {
		return false;
	}
	for (int i = 0; i < size; i++) {
		if (!(*dtype).equals(get(i), v.get(i))) {
			return false;
		}
	}
	return true;
}


string Array::toString() {
	ostringstream os;
	os << this;
	return os.str();
}

ostream& operator<<(ostream& out, Array& vec)
{
	out << vec.dtype->getName() << " Array [";
	for (unsigned int j = 0; j < vec.size; j++)
	{
		if (j % vec.size == vec.size - 1)
			out << vec.asString(j) << "]" << endl;
		else
			out << vec.asString(j) << ", ";
	}
	return out;
}
