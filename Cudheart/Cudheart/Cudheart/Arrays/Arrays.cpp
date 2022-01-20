#include "Arrays.h"

template <typename T>
Vector<T> Arrays::arange(double low, double high, double jump) {
	int size = (high - low) / jump;
	void* raw = operator new[](size * sizeof(T));
	T* ptr = static_cast<T*>(raw);

	for (int i = 0; i < size; ++i) {
		new(&ptr[i])T(i % 4, i / 4);
	}

	operator delete[](raw);

	for (double i = low; i < high; i += jump) {
		ptr[i] = i;
	}

	return Vector<T>(ptr, size);

}