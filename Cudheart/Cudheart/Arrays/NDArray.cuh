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

		virtual T get(int index) = 0;
		
		virtual void set(int index, T value) = 0;
		
		virtual NDArray<T>* emptyLike() = 0;
		
		virtual int getSize() = 0;
		
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
	};
}