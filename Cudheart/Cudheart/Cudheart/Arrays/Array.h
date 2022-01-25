#pragma once

#include "../Inc.h"
#include "../Dtypes/Dtype.h"
#include "Shape.h"

class Array {
public:
	/// <summary>
	/// the size of this vector
	/// </summary>
	int size;
	/// <summary>
	/// a pointer to this vector's data type
	/// </summary>
	Dtype *dtype;

	/// <summary>
	/// the array this vector contains
	/// </summary>
	void* arr;
	
	Shape* shape;
private:
	/// <summary>
	/// whether or not this vector was copied from somewhere.
	/// i need a better way to prevent double deletion
	/// </summary>
	bool copied = false;
public:
	/// <summary>
	/// create a new vector from a size. assumes dtype to be int.
	/// </summary>
	/// <param name="size"> : int - the size of the array to be created</param>
	Array(Shape *shape);
	Array(void* arr, Shape *shape);
	Array(Shape *shape, Dtype *dtype);
	Array(void* arr, Shape *shape, Dtype *dtype);

	~Array();

	string toString();

	friend ostream& operator<<(ostream& out, Array &v);

	void* operator[](size_t i);

	void* get(size_t i);
	void* get(int len, ...);

	bool operator==(Array &v);

	void set(size_t i, void* value);

	void setCopied(bool b) {
		copied = b;
	}

	string asString(size_t i);

	string getShapeString();

	void reshape(Shape* shape);
};