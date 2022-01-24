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

void* DDouble::copy(void* arr, Shape* shape)
{
	double* actual = (double*)arr;
	int size = (*shape).size;
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

bool DDouble::equals(void* a, void* b)
{
	double trueA = *(double*)a;
	double trueB = *(double*)b;
	return trueA == trueB;
}

void* DDouble::empty(Shape* shape)
{
	return (void*)(new double[(*shape).size]);
}
