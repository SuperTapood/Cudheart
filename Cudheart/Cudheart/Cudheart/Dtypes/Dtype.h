#pragma once

#include "../Inc.h"
#include "../Arrays/Shape.h"

class Dtype {
public:
	/// <summary>
	/// return a void pointer of the value in a specific index of a vector
	/// </summary>
	/// <param name="arr"> : void* - the array to get the value of</param>
	/// <param name="i"> : int - the index of the value</param>
	/// <returns> : void* - a pointer to the value fetched</returns>
	virtual void* get(void* arr, size_t i) = 0;
	/// <summary>
	/// convert a value of an array into a string based on this dtype
	/// </summary>
	/// <param name="arr"> : void* - the array to convert</param>
	/// <param name="i"> : int - the index of the value </param>
	/// <returns> string - the string representation of the value</returns>
	virtual string toString(void* arr, size_t i) = 0;
	/// <summary>
	/// get the name of this dtype
	/// </summary>
	/// <returns>string - the name of this dtype</returns>
	virtual string getName() = 0;
	/// <summary>
	/// return a copy of the given array, while converting both into this data type
	/// </summary>
	/// <param name="arr"> : void* - the array to copy</param>
	/// <param name="size"> : int - the size of the array</param>
	/// <returns> : void* - the copied array</returns>
	virtual void* copy(void* arr, Shape* shape) = 0;
	/// <summary>
	/// get the size of this data type
	/// </summary>
	/// <returns> : int - the size of the data type this object represents</returns>
	virtual int getSize() = 0;
	/// <summary>
	/// returns an empty (uninitialized) array of a chosen size
	/// </summary>
	/// <param name="size"> : int - the size of the empty thing</param>
	/// <returns> void* - the empty array</returns>
	virtual void* empty(Shape* shape) = 0;
	/// <summary>
	/// set the value at an index of an array to a particular. gets a void pointer bc cpp is fun
	/// </summary>
	/// <param name="arr"> : void* - the array to replace the value of</param>
	/// <param name="i"> : int - the index of the value to replace</param>
	/// <param name="value"> : void* - the void pointer of the value to replace</param>
	virtual void set(void* arr, size_t i, void* value) = 0;
	virtual bool equals(void* a, void* b) = 0;
};
