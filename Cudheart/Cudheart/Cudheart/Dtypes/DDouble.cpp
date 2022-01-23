#include "DDouble.h"

void* DDouble::get(void* arr, size_t i) {
	double* a = (double*)arr;
	return &a[i];
}

string DDouble::toString(void* arr, size_t i) {
	ostringstream os;
	os << *((double*)(get(arr, i)));
	return os.str();
}

string DDouble::getName() {
	return "Float64";
}

void* DDouble::copy(void* arr, int size)
{
	double* actual = (double*)arr;
	double* out = (double*)malloc(size * sizeof(int));

	for (int i = 0; i < size; i++) {
		out[i] = actual[i];
	}

	return (void*)out;
}

int DDouble::getSize()
{
	return sizeof(double);
}

void DDouble::set(void* arr, size_t i, void* value)
{
	double* actual = (double*)arr;
	actual[i] = *(double*)value;
}

void* DDouble::empty(int size)
{
	return (void*)(new double[size]);
}
