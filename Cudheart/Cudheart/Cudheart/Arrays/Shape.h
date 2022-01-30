#pragma once

#include "../Inc.h"


class Shape {
private:
	int* arr;
public:
	int length;
	int size;
public:
	Shape(Shape* shape);
	Shape(int x) : Shape({x}) {}
	Shape(int x, int y) : Shape({ x, y}) {}
	Shape(int x, int y, int z) : Shape({ x, y, z}) {}
	Shape(initializer_list<int> shape);
	int at(int index);
	string toString();
	Shape* dupe();
	int sizeFrom(int from);
};