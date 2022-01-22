#include "ArrayOps.h"
#include "../Inc.h"


VectorInt ArrayOps::asarray(int* arr, int size, Dtype dtype, bool copy)
{
	if (copy) {
		cout << size << endl;
		int* brr = new int[size];

		for (int i = 0; i < size; i++) {
			brr[i] = arr[i];
		}
		arr = brr;
	}
	return Dtype::asVector(arr, size);
}