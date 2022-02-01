#pragma once

#include "../Inc.h"
#include "Shape.h"
#include "../Exceptions/Exceptions.h"

template <typename T>
class Array {
public:
	/// <summary>
	/// the size of this vector
	/// </summary>
	int size;
private:
	/// <summary>
	/// whether or not this vector was copied from somewhere.
	/// i need a better way to prevent double deletion
	/// </summary>
	bool copied = false;

	/// <summary>
	/// the array this vector contains
	/// </summary>
	T* arr;

	Shape* shape;
public:
	/// <summary>
	/// create a new vector from a size. assumes dtype to be int.
	/// </summary>
	/// <param name="size"> : int - the size of the array to be created</param>
	Array(Shape* shape) {
		this->arr = (T*)malloc(size * sizeof(T));
		this->size = (*shape).size;
		this->shape = shape;
	}
	Array(T* arr, Shape *shape) {
		this->arr = arr;
		this->size = (*shape).size;
		this->shape = shape;
	}

	~Array()
	{
		/*delete arr;
		delete dtype;
		delete shape;*/
	}

	string asString(int i)
	{
		return to_string(arr[i]);
	}

	friend ostream& operator<<(ostream& out, Array<T>& vec)
	{
		out << vec.toString();
		return out;
	}

	T operator[](size_t i)
	{
		if (i < 0) {
			return this->operator[](size + i);
		}
		if (i > size) {
			throw IndexError(i, 0, size);
		}
		return arr[i];
	}

	T get(int len, ...)
	{
		// for now
		if (len != shape->length) {
			throw IndexError(len, shape);
		}
		int index = 0;
		va_list args;
		va_start(args, len);
		for (int i = 0; len > i; i++) {
			index += va_arg(args, int) * ((*shape).sizeFrom(i));
		}

		return operator[](index);
	}

	bool operator==(Array& v)
	{
		if (v.size != size) {
			return false;
		}
		for (int i = 0; i < size; i++) {
			if (arr[i] != (T)v.getFlat(i)) {
				return false;
			}
		}
		return true;
	}

	void setAbsolute(size_t i, T value)
	{
		arr[i] = value;
	}

	void set(T value, int len, ...)
	{
		// for now
		if (len != shape->length) {
			throw IndexError(len, shape);
		}
		int index = 0;
		va_list args;
		va_start(args, len);
		for (int i = 0; len > i; i++) {
			index += va_arg(args, int) * ((*shape).sizeFrom(i));
		}
	}

	void setCopied(bool b) {
		copied = b;
	}

	string toString() {
		int v = 0;
		int si = sizeof(int);
		T* arr = (T*)malloc((sizeof(T) * this->shape->length));
		for (int i = 0; i < this->shape->length; i++) {
			arr[i] = this->shape->at(i);
		}
		cout << "to string shape length: " << shape->length << endl;
		cout << "shape when to string: " << shape->toString() << endl;
		cout << "id to string: " << &shape << endl;
		string out = printRecursive(arr, shape->length, 0, 0);
		return out;
	}

	string getShapeString()
	{
		return shape->toString();
	}

	void reshape(Shape* shape)
	{
		if ((*shape).size != this->shape->size) {
			throw ShapeError(this->shape, shape);
		}
		delete this->shape;
		this->shape = shape;
	}

	T getFlat(int index) {
		return operator[](index);
	}

	Shape* dupeShape()
	{
		return shape->dupe();
	}

	int getDims()
	{
		return shape->length;
	}

	int getShapeAt(int index)
	{
		return shape->at(index);
	}

	Array<T>* dupe()
	{
		Shape s = shape->dupe();
		Array<T>* a = ArrayOps<T>::empty(&s);
		for (int i = 0; i < size; i++) {
			(*a).setAbsolute(i, getFlat(i));
		}
		cout << "dupe: " << (*a).toString() << endl;
		return a;
	}

private:
	string printRecursive(T* s, int len, int start, int offset)
	{
		cout << "id print recur: " << &len << endl;
		ostringstream os;
		os << "[";
		if (len == start + 1) {
			for (int i = 0; i < s[start]; i++) {
				if (i == 0) {
					os << arr[offset + i] << ",";
				}
				else {
					os << " " << arr[offset + i] << ",";
				}
			}
		}
		else {
			os << "\n";
			for (int i = 0; i < s[start]; i++) {
				for (int i = 0; i <= start; i++) {
					os << "  ";
				}
				os << printRecursive(s, len, start + 1, offset) << ",\n";
				offset += shape->sizeFrom(start);
			}
			for (int i = 0; i <= start; i++) {
				os << " ";
			}
		}
		os << "]";
		return os.str();
	}
};

#include "Array.cpp"