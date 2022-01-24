#pragma once

#include "../Inc.h"


class Shape {
private:
	int* arr;
public:
	int size;
	int length;
public:
	Shape(int x) : Shape({x}) {}
	Shape(int x, int y) : Shape({ x, y}) {}
	Shape(int x, int y, int z) : Shape({ x, y, z}) {}
	Shape(int* arr);
	Shape(initializer_list<int> shape);
	string toString();
	Shape* dupe();
};