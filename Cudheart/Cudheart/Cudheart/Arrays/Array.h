#pragma once

#include "Shape.h"
#include "../Exceptions/Exceptions.h"

namespace Cudheart::Arrays {
	template <typename T>
	class Array {
	public:
		T* data;
		Shape* shape;

	public:
		Array(Shape* shape) {
			this->shape = shape;
			print(shape->toString());
			data = (T*)malloc(shape->size * sizeof(T));
		}

		Array(int dims[], int len) {
			this->shape = new Shape(dims, len);
			data = (T*)malloc(shape->size * sizeof(T));
		}

		Array(T* arr, Shape* shape) {
			this->shape = shape;
			data = (T*)malloc(shape->size * sizeof(T));
			for (int i = 0; i < shape->size; i++) {
				data[i] = arr[i];
			}
		}

		Array* flatten() {
			return new Array<int>(data, new Shape(new int[] {this->shape->size}));
		}

		string toString() {
			return stringify(0, 0);
		}

		friend ostream& operator<<(ostream& out, Array<T>& arr)
		{
			out << arr.toString();
			return out;
		}

		Array<T>* reshape(Shape* newShape) {
			if (shape->size != newShape->size) {
				throw ShapeError(shape->toString(), newShape->toString());
			}
			return new Array<T>(data, newShape);
		}

	private:
		string stringify(int start, int offset) {
			ostringstream os;
			os << "[";
			if (start == shape->len - 1) {
				os << data[offset++];
				for (int i = 0; i < this->shape->dims[start]; offset++, i++) {
					os << ", " << data[offset];
				}
			} else {
				for (int j = 0; j < this->shape->dims[start]; j++, offset += this->shape->dims[start]) {
					os << "\n";
					for (int i = 0; i < start + 1; i++) {
						os << " ";
					}
					os << stringify(start + 1, offset);
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
}