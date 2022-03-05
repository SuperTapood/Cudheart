#include "Shape.h"

Cudheart::Arrays::Shape::Shape(int dims[], int len)
{
	this->dims = dims;
	this->len = len;
	
	size = 1;
	for (int i = 0; i < len; i++) {
		size *= dims[i];
	}
}

string Cudheart::Arrays::Shape::toString()
{
	ostringstream os;
	os << "(";
	for (int i = 0; i < len; i++) {
		os << dims[i] << ",";
	}
	os << ")";
	return os.str();
}
