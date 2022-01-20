#include "Vector.h"

template <typename T>
Vector<T>::Vector(T* arr, int size) {
	this->arr = arr;
	this->size = size;
}

template <typename T>
Vector<T>::Vector(T* arr) {
	this->arr = arr;
	size = sizeof(arr) / sizeof(arr[0]);
}

template <typename T>
Vector<T>::~Vector() {
	for (int i = size - 1; i > 0; i++) {
		arr[i].~T();
	}
}

template<typename T>
string Vector<T>::toString() {
	ostringstream os;
	os << arr << endl;
	return os.str();
}