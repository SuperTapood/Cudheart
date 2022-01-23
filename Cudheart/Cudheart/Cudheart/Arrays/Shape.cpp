#include "Shape.h"


Shape::Shape(initializer_list<int> shape) {
	size = shape.size();
	arr = (int*)malloc(size * sizeof(int));
	int i = 0;
	for (auto elem : shape)
	{
		arr[i++] = elem;
	}
}

string Shape::toString() {
	ostringstream os;
	os << "Shape(";
	for (int i = 0; i < size; i++) {
		os << arr[i] << ",";
	}
	os << ")";
	return os.str();
}