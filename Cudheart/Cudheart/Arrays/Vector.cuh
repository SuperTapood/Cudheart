#pragma once

#include "NDArray.cuh"
#include "../CUtil/CUtil.cuh"
#include "../Exceptions/Exceptions.cuh"
#include "Shape.cuh"

// check about using longs and stuff as lengths and indices for bigger tensors

namespace Cudheart::NDArrays {
	template <typename T>
	class Matrix;

	template <class T>
	class Vector : public NDArray<T> {
	private:
		int m_size;
		T* m_data;

	public:
		/// <summary>
		/// construct a new vector from the given data and size
		/// </summary>
		/// <param name="data"> - the given raw data</param>
		/// <param name="size"> - the given size of the raw data</param>
		Vector(T* data, int size) {
			m_data = data;
			m_size = size;
		}

		/// <summary>
		/// create a new empty vector with the given size
		/// </summary>
		/// <param name="size"> - the size of the vector</param>
		Vector(int size) {
			// m_data = (T*)malloc(size * sizeof(T));
			m_data = new T[size];
			m_size = size;
		}

		/// <summary>
		/// create a vector from the given raw data, and guess its size using pointer arithmetic. may or may not work.
		/// </summary>
		/// <param name="data"> - the provided raw data</param>
		Vector(T* data) {
			m_data = data;
			m_size = (&data)[1] - data;
		}

		/// <summary>
		/// create a vector from an initializer list
		/// </summary>
		/// <param name="list"> - the given initializer list</param>
		Vector(initializer_list<T> list) {
			m_size = list.size();
			m_data = new T[m_size];
			int i = 0;
			for (auto& x : list) {
				m_data[i] = x;
				i++;
			}
			shape = new Shape(m_size);
		}

		/// destroy the vector
		~Vector() {
			delete[] m_data;
		}

		/// <summary>
		/// cast this vector to another type
		/// </summary>
		/// <typeparam name="U"> - the type to cast to</typeparam>
		/// <returns>this vector but of U type</returns>
		template<typename U>
		Vector<U>* castTo() {
			Vector<U>* output = new Vector<U>(getSize());

			for (int i = 0; i < getSize(); i++) {
				output->set(i, (U)get(i));
			}

			return output;
		}

		/// <summary>
		/// get the element at the given index
		/// </summary>
		/// <param name="index"> - the index to get the element of</param>
		/// <returns>the element at the index position</returns>
		T get(int index) {
			if (index < 0) {
				index += m_size;
			}

			if (index >= m_size) {
				IndexOutOfBoundsException(m_size, index);
			}
			return m_data[index];
		}

		/// <summary>
		/// helper function for generalization
		/// </summary>
		/// <param name="index"> - the index of the row / column</param>
		/// <param name="axis"> - whether or not to use row (axis 0) or column (axis 1)</param>
		/// <returns>this vector dummy</returns>
		Vector<T>* get(int index, int axis) {
			return this;
		}

		/// <summary>
		/// set the element at index position to value
		/// </summary>
		/// <param name="index"> - the index of the element</param>
		/// <param name="value"> - the new value</param>
		void set(int index, T value) {
			if (index < 0) {
				index += m_size;
			}
			if (index >= m_size) {
				IndexOutOfBoundsException(m_size, index);
			}
			m_data[index] = value;
		}

		T getAbs(int index) {
			return get(index);
		}

		void setAbs(int index, T value) {
			return set(index, value);
		}

		template <typename U>
		NDArray<T>* emptyLike() {
			return new Vector<U>(m_size);
		}

		NDArray<T>* emptyLike() {
			return new Vector<T>(m_size);
		}

		template <typename U>
		NDArray<T>* shapeLike(Shape* other) {
			assertMatchShape(other);
			if (other->getDims() == 1) {
				return this;
			}
			else if (other->getDims() == 2) {
				Matrix<T>* out = new Matrix<T>(other->getY(), other->getX());

				for (int i = 0; i < other->getX(); i++) {
					for (int j = 0; j < other->getY(); j++) {
						out->set(i, j, get(i));
					}
				}

				return out;
			}

			return nullptr;
		}

		int getDims() {
			return 1;
		}

		/// <summary>
		/// get the size of the vector
		/// </summary>
		/// <returns>the size of the vector</returns>
		int getSize() {
			return m_size;
		}

		/// <summary>
		/// convert this vector to a string
		/// </summary>
		/// <returns>a string representation of this vector</returns>
		string toString() {
			ostringstream os;
			os << "[";
			for (int i = 0; i < getSize() - 1; i++) {
				os << m_data[i] << ", ";
			}
			os << get(-1) << "]";
			return os.str();
		}

		/// <summary>
		/// print this vector to the console
		/// </summary>
		void print() {
			cout << this->toString() << endl;
		}

		/// <summary>
		/// print info about this vector to the console
		/// </summary>
		void printInfo() {
			cout << "Vector of size: " << m_size << endl;
		}
		// todo: add operator overloades to make this look better

		/// <summary>
		/// get a cuda container containing this vector as a cuda array
		/// </summary>
		/// <returns>the resulting container</returns>
		ContainerA<T>* getContainerA() {
			ContainerA<T>* out = new ContainerA<T>();

			out->warmUp(m_data, m_size);

			return out;
		}

		/// <summary>
		/// get a cuda container containing this vector and another one as
		/// cuda arrays
		/// </summary>
		/// <param name="other"> - the other vector</param>
		/// <returns>the resulting container</returns>
		ContainerAB<T>* getContainerAB(Vector<T>* other) {
			ContainerAB<T>* out = new ContainerAB<T>();

			out->warmUp(m_data, other->m_data, m_size);

			return out;
		}

		/// <summary>
		/// get a cuda container containing this vector and two other ones as
		/// cuda arrays
		/// </summary>
		/// <param name="b"> - the second vector</param>
		/// <param name="c"> - the third vector</param>
		/// <returns></returns>
		ContainerABC<T>* getContainerABC(Vector<T>* b, Vector<T>* c) {
			ContainerABC<T>* out = new ContainerABC<T>();

			out->warmUp(m_data, b->m_data, c->m_data, m_size);

			return out;
		}

		/// <summary>
		/// assert that this vector matches another vector
		/// </summary>
		/// <param name="other"> - the other vector</param>
		void assertMatchShape(Shape* shape, int axis) {
			if (shape->getDims() == 1) {
				if (m_size != shape->getSize()) {
					Cudheart::Exceptions::ShapeMismatchException(m_size,
						shape->getSize()).raise();
				}
			}
			else if (shape->getDims() == 2) {
				if (axis == 0) {
					if (m_size != shape->getX()) {
						Cudheart::Exceptions::ShapeMismatchException(m_size,
							shape->getX()).raise();
					}
				}
				else if (axis == 1) {
					if (m_size != shape->getY()) {
						Cudheart::Exceptions::ShapeMismatchException(m_size,
							shape->getY()).raise();
					}
				}
			}
		}

		void assertMatchShape(Shape* shape) {
			return assertMatchShape(shape, 0);
		}

		Shape* getShape() {
			return new Shape(m_size);
		}

		NDArray<T>* copy() {
			Vector<T>* out = new Vector<T>(m_size);

			for (int i = 0; i < m_size; i++) {
				out->set(i, m_data[i]);
			}

			return out;
		}

		NDArray<T>* transpose() {
			return copy();
		}

		NDArray<T>* transpose(bool inplace) {
			if (inplace) {
				return this;
			}
			return copy();
		}

		NDArray<T>* flatten() {
			return copy();
		}
	};
}