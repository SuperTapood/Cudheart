#pragma once

#include "../Inc.h"


class Vector {
public:
	int size;
public:
	Vector(int size);
public:
	virtual string toString() {
		return "";
	}

	// virtual std::ostream& operator<<(std::ostream& out) = 0;

	// inherting classes need to implement:
	// (in order to be treated as actual usable vectors)
	// void* arr;
	// void operator[](std::size_t i);
	// void set(void value);
};