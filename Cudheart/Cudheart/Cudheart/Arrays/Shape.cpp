#include "Shape.h"

Shape::Shape(Shape* s) {
	length = s->length;
	size = s->size;
	arr = (int*)malloc(length * sizeof(int));
	for (int i = 0; i < length; i++) {
		arr[i] = s->arr[i];
	}
}

int Shape::at(int index)
{
	return arr[index];
}

string Shape::toString() {
	ostringstream os;
	os << "(";
	cout << "to string id: " << this << endl;
	cout << "this len shape string: " << this->length << endl;
	for (int i = 0; i < this->length; i++) {
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
	cout << "size from from: " << from << endl;
	cout << "size from length: " << &this->length << endl;
	cout << "id size from: " << this << endl;
	int out = 1;
	for (int i = from + 1; i < this->length; i++) {
		out *= arr[i];
	}
	return out;
}
