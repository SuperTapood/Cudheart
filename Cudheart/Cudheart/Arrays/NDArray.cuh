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
		virtual ~NDArray() {}

		/// <summary>
		/// get the element in the specified index
		/// </summary>
		/// <param name="index"> - the index</param>
		/// <returns>the element in the specified position</returns>
		virtual T get(int index) = 0;

		/// <summary>
		/// set the element in the specified index
		/// </summary>
		/// <param name="index"> - the index to set</param>
		/// <param name="value"> - the value to set to</param>
		virtual void set(int index, T value) = 0;

		/// <summary>
		/// convert this ndarray to a string
		/// </summary>
		/// <returns>a string representation of this ndarray object</returns>
		virtual string toString() = 0;

		/// <summary>
		/// print this ndarray object as a string
		/// </summary>
		virtual void print() = 0;
	};
}