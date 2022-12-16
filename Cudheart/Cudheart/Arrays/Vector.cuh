#pragma once

#include "NDArray.cuh"
#include "../Exceptions/Exceptions.cuh"
#include "Shape.cuh"

namespace Cudheart::NDArrays {
	template <typename T>
	class Matrix;

	template <class T>
	class Vector : public NDArray<T> {
	private:
		int m_size;
		T* m_data;

		string getString(int i) {
			if constexpr (is_same_v<T, StringType*>) {
				return ((StringType*)get(i))->toString();
			}
			else if constexpr (is_same_v<T, ComplexType*>) {
				return ((ComplexType*)get(i))->toString();
			}
			else if constexpr (std::is_fundamental<T>::value ) {
				return to_string(get(i));
			}

			return "";
		}

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
			constexpr bool isStringType = is_same_v<U, StringType*>;
			constexpr bool amStringType = is_same_v<T, StringType*>;
			constexpr bool isComplexType = is_same_v<U, ComplexType*>;
			constexpr bool amArithmetic = is_arithmetic_v<T>;
			constexpr bool isArithmetic = is_arithmetic_v<U>;
			constexpr bool isVoid = is_void_v<U>;
			constexpr bool isNull = is_null_pointer_v<U>;
			if (isStringType) {
				auto out = new Vector<StringType*>(getSize());

				for (int i = 0; i < getSize(); i++) {
					out->set(i, new StringType(getString(i)));
				}

				return (Vector<U>*)out;
			}
			if (isComplexType) {
				if (amArithmetic) {
					Vector<ComplexType*>* out = new Vector<ComplexType*>(getSize());

					for (int i = 0; i < getSize(); i++) {
						out->set(i, new ComplexType(get(i)));
					}

					return (Vector<U>*)out;
				}
			}
			if (isArithmetic) {
				if (amStringType) {
					Vector<U>* out = new Vector<U>(getSize());

					for (int i = 0; i < getSize(); i++) {
						auto str = (StringType*)get(i);
						out->set(i, (U)str->toFloating());
					}

					return out;
				}
			}
			if (isArithmetic) {
				if (!isVoid) {
					if (!isNull) {
						if (amArithmetic) {
							auto out = new Vector<U>(getSize());

							for (int i = 0; i < getSize(); i++) {
								// jumping through hoops to appease
								// our compiler overlord
								// lots of performance left on the table here
								out->set(i, std::stold(getString(i)));
							}

							return out;
						}
					}
				}
			}
			ostringstream os;
			os << "BadTypeException: cannot cast ";
			os << typeid(T).name();
			os << " type to ";
			os << typeid(U).name();
			os << " type.";
			BadTypeException(os.str());

			return NULL;
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

		template <typename T>
		NDArray<T>* shapeLike(Shape* other) {
			assertMatchShape(other);
			if (other->getDims() == 1) {
				return this;
			}
			else if (other->getDims() == 2) {
				Matrix<T>* out = new Matrix<T>(other->getX(), other->getY());

				for (int i = 0; i < out->getSize(); i++) {
					out->set(i, get(i));
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
			if constexpr (is_same_v<T, StringType*>) {
				for (int i = 0; i < getSize() - 1; i++) {
					os << ((StringType*)(m_data[i]))->toString() << ", ";
				}
				os << ((StringType*)(get(-1)))->toString() << "]";
			}
			else if constexpr (is_same_v<T, ComplexType*>) {
				for (int i = 0; i < getSize() - 1; i++) {
					os << ((ComplexType*)(m_data[i]))->toString() << ", ";
				}
				os << ((ComplexType*)(get(-1)))->toString() << "]";
			}
			else {
				for (int i = 0; i < getSize() - 1; i++) {
					os << m_data[i] << ", ";
				}
				os << get(-1) << "]";
			}
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

		void assertMatchShape(Shape* shape) {
			if (shape->getSize() != getSize()) {
				ostringstream os;
				os << "(";
				os << getSize();
				os << ",)";
				ShapeMismatchException(os.str(), shape->toString());
			}
		}
		
		void assertMatchShape(Shape* shape, int axis) {
			assertMatchShape(shape);
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