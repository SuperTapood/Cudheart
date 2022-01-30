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
	/*delete arr;
	delete dtype;
	delete shape;*/
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
	delete this->shape;
	this->shape = shape;
}

string Array::printRecursive(int* s, int len, int start, int offset)
{
	ostringstream os;
	os << "[";
	if (len == start + 1) {
		for (int i = 0; i < s[start]; i++) {
			if (i == 0) {
				os << dtype->toString(arr, offset + i) << ",";
			}
			else {
				os << " " << dtype->toString(arr, offset + i) << ",";
			}
		}
	}
	else {
		os << "\n";
		for (int i = 0; i < s[start]; i++) {
			for (int i = 0; i <= start; i++) {
				os << "  ";
			}
			os << printRecursive(s, len, start + 1, offset) << ",\n";
			offset += shape->sizeFrom(start);
		}
		for (int i = 0; i <= start; i++) {
			os << " ";
		}
	}
	os << "]";
	return os.str();
}

//void* Array::get(size_t i)
//{
//	return operator[](i);
//}

void* Array::getFlat(int index) {
	return operator[](index);
}

Shape* Array::dupeShape()
{
	return shape->dupe();
}

Dtype* Array::dupeDtype()
{
	return dtype->dupe();
}

int Array::getDims()
{
	return shape->length;
}

int Array::getShapeAt(int index)
{
	return shape->at(index);
}

Array* Array::dupe()
{
	Shape s = shape->dupe();
	void* a = dtype->empty(&s);
	for (int i = 0; i < size; i++) {
		dtype->set(a, i, getFlat(i));
	}
	Array* out = new Array(a, &s, dtype->dupe());
	return out;
}

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

void Array::set(void* value, int len, ...)
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
		if (!(*dtype).equals(getFlat(i), v.getFlat(i))) {
			return false;
		}
	}
	return true;
}


string Array::toString() {
	int v = 0;
	int si = sizeof(int);
	void* a = malloc((sizeof(int) * shape->length));
	int* arr = (int*)a;
	for (int i = 0; i < shape->length; i++) {
		arr[i] = shape->at(i);
	}
	string out = printRecursive(arr, shape->length, 0, 0);
	return out;
}

ostream& operator<<(ostream& out, Array& vec)
{
	out << vec.toString();
	return out;
}
