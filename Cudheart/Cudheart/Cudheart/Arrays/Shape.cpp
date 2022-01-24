#include "Shape.h"


Shape::Shape(int* arr)
{
	this->arr = arr;
}

Shape::Shape(initializer_list<int> shape) {
	length = shape.size();
	arr = (int*)malloc(size * sizeof(int));
	int i = 0;
	size = 1;
	for (int elem : shape)
	{
		size *= elem;
		arr[i++] = elem;
	}
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
	return new Shape(arr);
}
