#pragma once

#include "../Util.cuh"

class Shape {
private:
	int* data;
	int dims;

public:
	Shape(int data[], int size);
};