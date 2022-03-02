#pragma once

#include "../../Util.h"
#include "Shape.h"


template <typename T>
class Array {
public:
	T* data;
	int size;
	Shape* shape;

public:
	Array(T* arr, initializer_list<int> list) {
		shape = new Shape(list);
		size = shape->size;
		data = (T*)malloc(size * sizeof(T));
		for (int i = 0; i < size; i++) {
			data[i] = arr[i];
		}
	}

	friend ostream& operator<<(ostream& out, Array<T>& vec)
	{
		out << vec.toString();
		return out;
	}

	string toString() {
		int v = 0;
		int si = sizeof(int);
		int* arr = (int*)malloc((sizeof(int) * this->shape->dims));
		for (int i = 0; i < this->shape->dims; i++) {
			arr[i] = this->shape->at(i);
		}
		string out = printRecursive(arr, shape->dims, 0, 0);
		return out;
	}

	~Array() {
		delete[] data;
	}



private:
	string printRecursive(T* shape, int len, int start, int offset)
	{
		cout << start << endl;
		ostringstream os;
		os << "[";
		os << "]";
		return os.str();
	}
};