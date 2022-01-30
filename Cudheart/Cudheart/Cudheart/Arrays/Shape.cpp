#include "Shape.h"

Shape::Shape(Shape* s) {
	length = s->length;
	size = s->size;
	arr = (int*)malloc(length * sizeof(int));
	for (int i = 0; i < length; i++) {
		arr[i] = s->arr[i];
	}
}

Shape::Shape(initializer_list<int> shape) {
	length = shape.size();
	arr = (int*)malloc(length * sizeof(int));
	int i = 0;
	size = 1;
	for (int elem : shape)
	{
		size *= elem;
		arr[i++] = elem;
	}
}

int Shape::at(int index)
{
	return arr[index];
}

string Shape::toString() {
	ostringstream os;
	os << "(";
	for (int i = 0; i < length; i++) {
		os << arr[i] << ",";
	}
	os << ")";
	return os.str();
}

Shape* Shape::dupe()
{
	return new Shape(this);
}

int Shape::sizeFrom(int from)
{
	int out = 1;
	for (int i = from + 1; i < length; i++) {
		out *= arr[i];
	}
	return out;
}
