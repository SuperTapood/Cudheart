#pragma once


#include "Dtype.h"

class DInt : public Dtype {
public:
	/// <summary>
	/// return a void pointer of the value in a specific index of a vector
	/// </summary>
	/// <param name="arr"> : void* - the array to get the value of</param>
	/// <param name="i"> : int - the index of the value</param>
	/// <returns> : void* - a pointer to the value fetched</returns>
	void* get(void* arr, size_t i) override;
	/// <summary>
	/// convert a value of an array into a string based on this dtype
	/// </summary>
	/// <param name="arr"> : void* - the array to convert</param>
	/// <param name="i"> : int - the index of the value </param>
	/// <returns> string - the string representation of the value</returns>
	string toString(void* arr, size_t i) override;
	/// <summary>
	/// get the name of this dtype
	/// </summary>
	/// <returns>string - the name of this dtype</returns>
	string getName() override;
	/// <summary>
	/// return a copy of the given array, while converting both into this data type
	/// </summary>
	/// <param name="arr"> : void* - the array to copy</param>
	/// <param name="size"> : int - the size of the array</param>
	/// <returns> : void* - the copied array</returns>
	void* copy(void* arr, int size) override;
	/// <summary>
	/// get the size of this data type
	/// </summary>
	/// <returns> : int - the size of the data type this object represents</returns>
	int getSize() override;
	/// <summary>
	/// returns an empty (uninitialized) array of a chosen size
	/// </summary>
	/// <param name="size"> : int - the size of the empty thing</param>
	/// <returns> void* - the empty array</returns>
	void* empty(int size) override;
	/// <summary>
	/// set the value at an index of an array to a particular. gets a void pointer bc cpp is fun
	/// </summary>
	/// <param name="arr"> : void* - the array to replace the value of</param>
	/// <param name="i"> : int - the index of the value to replace</param>
	/// <param name="value"> : void* - the void pointer of the value to replace</param>
	void set(void* arr, size_t i, void* value) override;
};