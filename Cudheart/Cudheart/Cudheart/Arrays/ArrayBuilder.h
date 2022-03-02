#pragma once
#include "Array.h"


template <typename T>
class ArrayBuilder {
	friend class Array<T>;
	
public:
	static Array<T>* asArray(T* data, int len) {
		return new Array<T>(data, len);
	}

	static Array<T>* empty(initializer_list<int> shape) {
		Shape* shap = new Shape(shape);
		T* empty = (T*)malloc(sizeof(T) * shap->size);
		delete shap;
		return new Array<T>(empty, shape);
	}
};