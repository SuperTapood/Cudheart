#pragma once

#include "../Inc.h"

class Dtype {
public:
	virtual void* get(void* arr, size_t i) = 0;
	virtual string toString(void* arr, size_t i) = 0;
	virtual string getName() = 0;
};
