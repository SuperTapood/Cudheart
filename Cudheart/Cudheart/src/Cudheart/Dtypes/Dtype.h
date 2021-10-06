#pragma once

#include "../Core/Shape.h"



// struct?

class Dtype {
public:
	void* cast(void* arr, int length, bool copy);
	void* cast(void* arr, Shape shape, bool copy);
};