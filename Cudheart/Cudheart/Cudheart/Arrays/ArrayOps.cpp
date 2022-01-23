#include "ArrayOps.h"
#include "../Inc.h"


Vector ArrayOps::asarray(int* arr, int size, Dtype *dtype, bool copy)
{
	if (copy) {
		int* brr = new int[size];
		for (int i = 0; i < size; i++) {
			brr[i] = arr[i];
		}
		arr = brr;
	}
	return Vector((void*)arr, size);
}