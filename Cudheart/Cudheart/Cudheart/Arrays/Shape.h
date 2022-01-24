#pragma once

#include "../Inc.h"


class Shape {
private:
	int* arr;
public:
	int size;
	int length;
public:
	Shape(initializer_list<int> shape);
	string toString();
};