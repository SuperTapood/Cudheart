#pragma once

#include "../Util.cuh"

class Shape {
private:
	int* data;
	int dims;
	int elems;

public:
	Shape(int* data, int size);
	int getDims();
	int getSize();
	int get(int index);
	Shape* clone();
	string toString();
};