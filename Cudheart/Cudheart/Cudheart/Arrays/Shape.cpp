#include "Shape.h"

Shape::Shape(initializer_list<int> list)
{
	dims = list.size();
	size = 1;
	shape = (int*)malloc(sizeof(int) * size);
	int i = 0;
	for (int l : list) {
		shape[i++] = l;
		size *= l;
	}
}

int Shape::sizeFrom(int from)
{
	int out = 1;
	for (int i = from + 1; i < this->dims; i++) {
		out *= shape[i];
	}
	return out;
}

int Shape::at(int index) {
	return shape[index];
}

string Shape::toString() {
	ostringstream os;
	os << "(";
	for (int i = 0; i < this->dims; i++) {
		os << shape[i] << ",";
	}
	os << ")";
	return os.str();
}
