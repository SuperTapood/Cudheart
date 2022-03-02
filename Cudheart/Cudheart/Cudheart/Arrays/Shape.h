#pragma once

#include "../../Util.h"


class Shape {
public:
	int* shape;
	int size;
	int dims;

public:
	Shape(initializer_list<int> list);
	int at(int index);
	string toString();
	int sizeFrom(int from);
};