#pragma once

#include "../Util.cuh"

namespace Cudheart::NDArrays {
	/// <summary>
	/// the base ndarray class for both the vector and the matrix
	/// </summary>
	/// <typeparam name="T"> - the type of the ndarray</typeparam>
	template <typename T>
	class NDArray {
	public:
		/// <summary>
		/// destroy this object
		/// </summary>
		~NDArray() {
			// std::cout << "ndarray destructor called" << std::endl;
		}

		/// <summary>
		/// get the element at the given index
		/// </summary>
		/// <param name="index"> - the index</param>
		/// <returns>the element at index position</returns>
		virtual T get(int index) = 0;
		
		/// <summary>
		/// set the element at the given index to given value
		/// </summary>
		/// <param name="index"> - the index</param>
		/// <param name="value"> - the value</param>
		virtual void set(int index, T value) = 0;
		
		/// <summary>
		/// return an empty object of the same type as this (vector or matrix)
		/// </summary>
		/// <returns>the resulting object as an ndarray</returns>
		virtual NDArray<T>* emptyLike() = 0;
		
		/// <summary>
		/// get the number elements this ndarray contains
		/// </summary>
		/// <returns></returns>
		virtual int getSize() = 0;
		
		/// <summary>
		/// get the number of dimensions this ndarray has (1 for vector, 2 for matrix)
		/// </summary>
		/// <returns></returns>
		virtual int getDims() = 0;

		/// <summary>
		/// convert this ndarray to a string
		/// </summary>
		/// <returns>a string representation of this ndarray object</returns>
		virtual string toString() = 0;

		/// <summary>
		/// print this ndarray object as a string
		/// </summary>
		virtual void print() = 0;

		virtual void assertMatchShape(NDArray<T>* arr, int axis) = 0;

		virtual void assertMatchShape(NDArray<T>* arr) = 0;
	};
}