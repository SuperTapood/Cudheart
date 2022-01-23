#include "DInt.h"

void* DInt::get(void* arr, size_t i) {
	int* a = (int*)arr;
	return &a[i];
}

string DInt::toString(void* arr, size_t i) {
	ostringstream os;
	os << *((int*)(get(arr, i)));
	return os.str();
}

string DInt::getName() {
	return "Int32";
}

void* DInt::copy(void* arr, int size) {
	int* actual = (int*)arr;
	int* out = (int*)malloc(size * sizeof(int));

	for (int i = 0; i < size; i++) {
		out[i] = actual[i];
	}

	return (void*)out;
}

int DInt::getSize()
{
	return sizeof(int);
}

void DInt::set(void* arr, size_t i, void* value)
{
	int* actual = (int*)arr;
	actual[i] = *(int*)value;
}

void* DInt::empty(int size)
{
	return (void*)(new int[size]);
}
