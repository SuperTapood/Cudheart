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

void* DInt::copy(void* arr, Shape* shape) {
	int* actual = (int*)arr;
	int size = (*shape).size;
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

bool DInt::equals(void* a, void* b)
{
	int trueA = *(int*)a;
	int trueB = *(int*)b;
	return trueA == trueB;
}

void* DInt::empty(Shape* shape)
{
	return (void*)(new int[(*shape).size]);
}
