#pragma once

#include "../Util.cuh"
#include "Shape.cuh"


template <typename T>
class Array {
private:
	T* data;
	Shape* shape;
	int offset;
	
public:
	Array(Shape* shape) {
		this->data = (T*)malloc(shape->getSize() * sizeof(T));
		this->shape = shape;
	}

	Array(T* data, Shape* shape) {
		this->data = data;
		this->shape = shape;
	}

	string toString() {
		offset = 0;
		return stringify(0);
	}

	void setAbs(int index, T value) {
		data[index] = value;
	}

	Array<T>* reshape(Shape* nshape) {
		if (nshape->getSize() != shape->getSize()) {
			return NULL;
		}
		return new Array<T>(data, nshape->clone());
	}

	void print() {
		cout << toString() << endl;
	}

	Shape* cloneShape() {
		return shape->clone();
	}

	int getDims() {
		return shape->getDims();
	}

	int getShape(int index) {
		return shape->get(index);
	}

	T getAbs(int index) {
		return data[index];
	}

	template <typename H>
	Array<T> operator&(const Array<T>& a, const Array<H> b) {
		// generate a higher dimensional array of both the arrays
	}

private:
	string stringify(int start) {
		ostringstream os;
		os << "[";
		if (start == shape->getDims() - 1) {
			os << data[offset++];
			for (int i = 1; i < this->shape->get(start);i++) {
				os << ", " << data[offset + i - 1];
			}
		}
		else {
			for (int j = 0; j < this->shape->get(start); j++, offset += this->shape->get(start)) {
				os << "\n";
				for (int i = 0; i < start + 1; i++) {
					os << " ";
				}
				os << stringify(start + 1);
			}
			os << "\n";
			for (int i = 0; i < start; i++) {
				os << " ";
			}
		}
		os << "]";
		if (start != 0) {
			os << ",";
		}
		return os.str();
	}
};