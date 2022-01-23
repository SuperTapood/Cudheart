#pragma once

#include "../Inc.h"
#include "../Dtypes/Dtype.h"

class Vector {
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
private:
	/// <summary>
	/// whether or not this vector was copied from somewhere.
	/// i need a better way to prevent double deletion
	/// </summary>
	bool copied;
public:
	/// <summary>
	/// create a new vector from a size. assumes dtype to be int.
	/// </summary>
	/// <param name="size"> : int - the size of the array to be created</param>
	Vector(int size);

	Vector(void* arr, int size);

	Vector(void* arr, int size, Dtype *dtype);

	~Vector();

	string toString();

	friend ostream& operator<<(ostream& out, Vector &v);

	void* operator[](size_t i);

	void set(size_t i, void* value);

	void setCopied(bool b) {
		copied = b;
	}

	string asString(size_t i);
};