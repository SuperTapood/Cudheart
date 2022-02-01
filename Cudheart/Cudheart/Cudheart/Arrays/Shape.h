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
	Shape(int x) {
		this->arr = (int*)malloc(sizeof(int));
		this->arr[0] = x;
		length = 1;
		size = x;
	}
	Shape(int x, int y) {
		this->arr = (int*)malloc(sizeof(int) * 2);
		this->arr[0] = x;
		this->arr[1] = y;
		length = 2;
		size = x * y;
	}
	Shape(int x, int y, int z) {
		this->arr = (int*)malloc(sizeof(int) * 3);
		this->arr[0] = x;
		this->arr[1] = y;
		this->arr[2] = z;
		length = 3;
		size = x * y * z;
	}
	// Shape(initializer_list<int> shape);
	int at(int index);
	string toString();
	Shape* dupe();
	int sizeFrom(int from);
	friend ostream& operator<<(ostream& out, Shape& s)
	{
		out << s.toString();
		return out;
	}
};