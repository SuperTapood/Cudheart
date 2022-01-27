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
	static Array v = empty(new Shape({size}), dtype);
	v.setCopied(true);

	for (int i = 0; i < v.size; i++) {
		v.setAbsolute(i, &low);
		low += jump;
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
	for (int i = 0; k < rows && i < cols; k++, i++) {
		cout << "k " << k << " i " << i << endl;
		out.setAbsolute(&one, 2, k, i);
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


