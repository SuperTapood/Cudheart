#include "Shape.cuh"


Shape::Shape(int* data, int size) {
	this->data = data;
	dims = size;
	elems = 1;
	for (int i = 0; i < dims; i++) {
		elems *= data[i];
	}
}

int Shape::getDims() {
	return dims;
}

int Shape::getSize() {
	return elems;
}

int Shape::get(int index) {
	return data[index];
}

Shape* Shape::clone() {
	int* copy = (int*)malloc(sizeof(int) * dims);
	for (int i = 0; i < dims; i++) {
		copy[i] = data[i];
	}
	return new Shape(copy, dims);
}

string Shape::toString() {
	ostringstream os;
	os << "(";
	for (int i = 0; i < dims; i++) {
		os << data[i] << ",";
	}
	os << ")";
	return os.str();
}