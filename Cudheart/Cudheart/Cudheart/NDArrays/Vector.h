#pragma once

#include "NDim.h"


template <class T>
class Vector : public NDim {
private:
	T* arr;
	void* raw_data;
public:
	int size;
public:
	Vector(void* raw_data, T* arr) {
		raw_data = operator new[](size* sizeof(T));
		arr = static_cast<T*>(raw_data);
	}

	~Vector() {
		for (int i = size - 1; i > 0; i--) {
			arr[i].~T();
		}
		operator delete[](raw_data);
	}
};