#include "Dint.h"

void* DInt32::get(void* arr, size_t i) {
	int* a = (int*)arr;
	return &a[i];
}

string DInt32::toString(void* arr, size_t i) {
	ostringstream os;
	os << *((int*)(get(arr, i)));
	return os.str();
}

string DInt32::getName() {
	return "Int32";
}