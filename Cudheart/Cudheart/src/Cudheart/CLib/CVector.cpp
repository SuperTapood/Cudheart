#include "CVector.h"
#include <stdio.h>



namespace Cudheart::CObjects {
	template <class T>
	CVector<T>::CVector(T arr[], Shape shape, Dtype dtype, bool copy) {
		init(arr, shape, dtype, copy);
	}

	template <class T>
	CVector<T>::CVector(T arr[], Shape shape, Dtype dtype)
	{
		init(arr, shape, dtype, true);
	}

	template <class T>
	CVector<T>::~CVector()
	{
		delete[] ptr;
	}

	template <class T>
	void CVector<T>::init(T arr[], Shape shape, Dtype dtype, bool copy)
	{
		if (shape.length > 0) {
			printf("shape with more than 1 dimension given. using the first dimension.");
		}
		ptr = new T[shape.size];
		for (int i = 0; i < shape.size; i++) {
			ptr[i] = arr[i];
		}
		if (!copy) {
			// pretend that we have moved T
			// noone has to know
			// it'll be our secret ;)
			delete[] T;
		}
		this->shape = shape;
		this->dtype = dtype;
	}
}
