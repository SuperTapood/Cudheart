#include "pch.h"
#include "ArrayOps.h"
#include "../Inc.h"


Array ArrayOps::asarray(void* arr, Shape *shape, Dtype *dtype)
{
	arr = dtype->copy(arr, shape);
	return Array((void*)arr, shape, dtype);
}

Array ArrayOps::arange(double low, double high, double jump, Dtype *dtype)
{
	int size = (high - low) / jump;
	Array out = empty(new Shape({size}), dtype);
	out.setCopied(true);
	// god knows why i need a temp variable but yeah
	double t = low;
	double* pt = &t;
	void* vt = (void*)pt;
	for (int i = 0; i < out.size; i++) {
		out.setAbsolute(i, vt);
		t += jump;
	}

	return out;
}

Array ArrayOps::empty(Shape* shape, Dtype* dtype)
{
	return asarray((void*)dtype->empty(shape), shape, dtype);
}

Array ArrayOps::eye(int rows, int cols, int k, Dtype* dtype)
{
	Array out = zeros(new Shape(rows, cols));
	
	int one = 1;
	for (int i = 0; k < cols && i < rows; k++, i++) {
		k = k % cols;
		out.set(&one, 2, i, k);
	}

	return out;
}

Array ArrayOps::full(Shape* shape, void* value, Dtype* dtype)
{
	Array out = empty(shape, dtype);
	for (int i = 0; i < out.size; i++) {
		out.setAbsolute(i, value);
	}
	return out;
}

Array* ArrayOps::meshgrid(Array* a, Array* b)
{
	if ((*a).getDims() != 1 || (*b).getDims() != 1) {
		return nullptr;
	}
	int alen = (*a).getShapeAt(0);
	int blen = (*b).getShapeAt(0);
	static Array* out = (Array*)malloc(sizeof(Array) * 2);
	out[0] = zeros(new Shape(blen, alen));
	out[1] = zeros(new Shape(blen, alen));
	cout << out[0].getShapeString() << endl;
	cout << out[1].getShapeString() << endl;

	for (int i = 0; i < blen; i++) {
		for (int j = 0; j < alen; j++) {
			out[0].set((*a).getFlat(j), 2, i, j);
		}
	}

	for (int i = 0; i < alen; i++) {
		for (int j = 0; j < blen; j++) {
			out[1].set((*b).getFlat(j), 2, j, i);
		}
	}

	return out;
}

Array ArrayOps::tril(Array* arr, int k)
{
	if ((*arr).getDims() != 2) {
		return nullptr;
	}
	int zero = 0;
	Array out = *(*arr).dupe();
	for (int i = 0; i < out.getShapeAt(0); i++) {
		for (int j = i + 1 + k; j < out.getShapeAt(1); j++) {
			out.set(&zero, 2, i, j);
		}
	}
	return out;
}


