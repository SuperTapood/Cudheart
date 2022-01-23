#pragma once

#include "../Inc.h"

class Dtype {
public:
	virtual void* get(void* arr, size_t i) = 0;
	virtual string toString(void* arr, size_t i) = 0;
	virtual string getName() = 0;
	virtual void* copy(void* arr, int size) = 0;
	virtual int getSize() = 0;
	virtual void* empty(int size) = 0;
	virtual void set(void* arr, size_t i, void* value) = 0;
};
