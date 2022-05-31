#pragma once

#include "../Util.cuh"

namespace Cudheart::NDArrays {
	class Shape {
	private:
		int x, y;
		int size;
		int dims;

	public:
		Shape(int x, int y) {
			this->x = x;
			this->y = y;
			size = x * y;
			this->dims = 2;
		}

		Shape(int x) {
			this->x = x;
			this->y = x;
			size = x;
			this->dims = 1;
		}

		int getX() {
			return x;
		}

		int getY() {
			return y;
		}

		int getDims() {
			return dims;
		}

		int getSize() {
			return size;
		}
	};
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
	};
}