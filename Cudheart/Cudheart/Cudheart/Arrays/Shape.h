#pragma once

#include "../Inc.h"


class Shape {
private:
	int* arr;
public:
	int size;
public:
	Shape(initializer_list<int> shape);
	string toString();
};