#pragma once

#include "../Util.cuh"
#include "Shape.cuh"
#include "../StringTypes/StringType.cuh"
#include "../Math/Complex/ComplexType.cuh"
#include "../Exceptions/BadTypeException.cuh"

namespace Cudheart::NDArrays {
	/// <summary>
	/// the base ndarray class for both the vector and the matrix
	/// </summary>
	/// <typeparam name="T"> - the type of the ndarray</typeparam>
	template <typename T>
	class NDArray {
	public:
		NDArray() {
			if constexpr (std::is_void_v<T>) {
				if constexpr (std::is_null_pointer_v<T>) {
					ostringstream os;
					os << "BadTypeException: cannot create an ndarray of type ";
					os << typeid(T).name();
					os << ".";
					BadTypeException(os.str());
				}
			}
		}
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
		
		virtual NDArray<T>* emptyLike() = 0;

		virtual NDArray<T>* transpose() = 0;

		virtual NDArray<T>* transpose(bool inplace) = 0;

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

		virtual void assertMatchShape(Shape* other, int axis) = 0;

		virtual void assertMatchShape(Shape* other) = 0;

		virtual Shape* getShape() = 0;

		virtual NDArray<T>* copy() = 0;

		virtual NDArray<T>* flatten() = 0;

		string getString(int index) {
			if constexpr (std::is_same_v<T, StringType*>) {
				return ((StringType*)(get(index)))->toString();
			}
			else if constexpr (std::is_same_v<T, ComplexType*>) {
				return ((ComplexType*)(get(index)))->toString();
			}
			else {
				return std::to_string(get(index));
			}
			return "";
		}
	};
}