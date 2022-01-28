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
	Array v = empty(new Shape({size}), dtype);
	v.setCopied(true);
	// god knows why i need a temp variable but yeah
	double t = low;
	double* pt = &t;
	void* vt = (void*)pt;
	for (int i = 0; i < v.size; i++) {
		v.setAbsolute(i, vt);
		t += jump;
	}

	return v;
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
	static Array* out = (Array*)malloc(sizeof(Array) * 2);
	out[0] = zeros(new Shape((*a).getShapeAt(0), (*b).getShapeAt(0)));
	out[1] = zeros(new Shape((*b).getShapeAt(0), (*a).getShapeAt(0)));
	for (int i = 0; i < (*b).getShapeAt(0); i++) {
		for (int j = 0; j < (*a).getShapeAt(0); j++) {
			cout << "i " << i << " j " << j << " value " << *(int*)(*a).getFlat(i) << endl;
			out[0].set((*a).getFlat(i), 2, j, i);
		}
	}
	for (int i = 0; i < (*b).getShapeAt(0); i++) {
		for (int j = 0; j < (*a).getShapeAt(0); j++) {
			out[1].set((*b).getFlat(i), 2, i, j);
		}
	}
	return out;
}


