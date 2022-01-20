#pragma once

#include <string>
#include <ostream>
#include <iostream>

using std::string;
using std::ostringstream;
using std::endl;


template <typename T>
class Vector {
private:
	T* arr;
public:
	int size;
public:
	Vector(T* arr, int size);

	Vector(T* arr);

	~Vector();

	string toString();
};