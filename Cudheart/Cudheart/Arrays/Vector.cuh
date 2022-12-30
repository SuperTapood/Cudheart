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
		Shape* m_shape = nullptr;

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
			m_data = new T[size];
			m_size = size;
		}

		/*/// <summary>
		/// create a vector from the given raw data, and guess its size using pointer arithmetic. may or may not work.
		/// </summary>
		/// <param name="data"> - the provided raw data</param>
		Vector(T* data) {
			m_data = data;
			m_size = ????
		}*/

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
		// even god himself can't help this function
		template<typename U>
		Vector<U>* castTo() {
			if constexpr (std::is_same_v<T, U>) {
				return this;
			}
			if constexpr (std::is_same_v<U, StringType*>) {
				// Convert the vector of T to a vector of StringType*.
				// For example, if T is a numeric type, then each element of the vector
				// will be converted to a string representation.
				Vector<StringType*>* out = new Vector<StringType*>(getSize());
				for (int i = 0; i < getSize(); i++) {
					out->set(i, new StringType(getString(i)));
				}
				return (Vector<U>*)out;
			}
			else if constexpr (std::is_same_v<U, ComplexType*>) {
				if constexpr (std::is_arithmetic_v<T>) {
					// Convert the vector of T to a vector of ComplexType*.
					// For example, if T is a numeric type, then each element of the vector
					// will be wrapped in a ComplexType object.
					Vector<ComplexType*>* out = new Vector<ComplexType*>(getSize());
					for (int i = 0; i < getSize(); i++) {
						out->set(i, new ComplexType(get(i)));
					}
					return (Vector<U>*)out;
				}
				else if constexpr (std::is_same_v<T, StringType*>) {
					// Convert the vector of T to a vector of ComplexType*.
					// For example, if T is a numeric type, then each element of the vector
					// will be wrapped in a ComplexType object.
					Vector<ComplexType*>* out = new Vector<ComplexType*>(getSize());
					for (int i = 0; i < getSize(); i++) {
						string current = getString(i);
						int pos = current.find("+");
						if (current.find("j") == string::npos || pos == string::npos) {
							bool isInt = true;
							for (int i = 0; i < current.size() && isInt; i++) {
								isInt = isdigit(current[i]);
							}

							if (!isInt) {
								std::ostringstream os;
								os << "BadTypeException: cannot convert ";
								os << current;
								os << " to a Complex type.";
								throw BadTypeException(os.str());
							}
							out->set(i, new ComplexType(std::stold(getString(i))));
						}
						else {
							bool isInt = true;
							for (int i = 0; i < pos && isInt; i++) {
								isInt = isdigit(current[i]);
							}

							if (!isInt) {
								std::ostringstream os;
								os << "BadTypeException: cannot convert ";
								os << current;
								os << " to a Complex type.";
								throw BadTypeException(os.str());
							}

							pos++;

							for (int i = pos; i < current.size() - 1 && isInt; i++) {
								isInt = isdigit(current[i]);
							}

							if (!isInt) {
								std::ostringstream os;
								os << "BadTypeException: cannot convert ";
								os << current;
								os << " to a Complex type.";
								throw BadTypeException(os.str());
							}

							out->set(i, new ComplexType(std::stold(getString(i))));
						}
					}
					return (Vector<U>*)out;
				}
			}
			else if constexpr (std::is_arithmetic_v<U>) {
				if constexpr (std::is_same_v<T, StringType*>) {
					// Convert the vector of StringType* to a vector of U.
					// For example, if U is a numeric type, then each element of the vector
					// will be converted from a string representation to a numeric value.
					Vector<U>* out = new Vector<U>(getSize());
					for (int i = 0; i < getSize(); i++) {
						auto str = (StringType*)get(i);
						out->set(i, (U)(str->toFloating()));
					}
					return out;
				}
				else if constexpr (std::is_arithmetic_v<T>) {
					// Convert the vector of T to a vector of U.
					// For example, if T and U are both numeric types, then each element of the vector
					// will be converted from one type to the other.
					Vector<U>* out = new Vector<U>(getSize());
					for (int i = 0; i < getSize(); i++) {
						out->set(i, static_cast<U>(get(i)));
					}
					return out;
				}
			}

			// If none of the above conditions are met, then it is not possible to perform the conversion.
			std::ostringstream os;
			os << "BadTypeException: cannot cast ";
			os << typeid(T).name();
			os << " type to ";
			os << typeid(U).name();
			os << " type.";
			throw BadTypeException(os.str());
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

			if (index >= m_size || index < 0) {
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

		// why tf are these functions needed
		template <typename U>
		NDArray<T>* emptyLike() {
			return new Vector<U>(m_size);
		}

		NDArray<T>* emptyLike() {
			return new Vector<T>(m_size);
		}

		template <typename U>
		NDArray<U>* reshape(Shape* shape) {
			if (shape->getDims() == 2) {
				return asMatrix<U>(shape->getX(), shape->getY());
			}

			if (shape->getX() != m_size) {
				throw ShapeMismatchException(to_string(m_size), to_string(shape->getX()));
			}

			return castTo<U>();
		}

		template <typename U>
		Matrix<U>* asMatrix(int x, int y) {
			if (x * y != m_size) {
				ostringstream os;
				os << "(";
				os << x;
				os << ",";
				os << y;
				os << ")";
				throw ShapeMismatchException(to_string(m_size), os.str());
			}

			Matrix<U>* out = new Matrix<U>(x, y);

			for (int i = 0; i < m_size; i++) {
				out->set(i, (U)get(i));
			}

			return out;
		}

		inline int getDims() const {
			return 1;
		}

		/// <summary>
		/// get the size of the vector
		/// </summary>
		/// <returns>the size of the vector</returns>
		int getSize() const {
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

		void assertMatchShape(Shape* shape, int axis = 0) {
			if (shape->getSize() != getSize()) {
				ostringstream os;
				os << "(";
				os << getSize();
				os << ",)";
				ShapeMismatchException(os.str(), shape->toString());
			}
		}

		Shape* getShape() {
			if (m_shape == nullptr) {
				m_shape = new Shape(m_size);
			}
			return m_shape;
		}

		NDArray<T>* copy() {
			Vector<T>* out = new Vector<T>(m_size);

			for (int i = 0; i < m_size; i++) {
				out->set(i, m_data[i]);
			}

			return out;
		}

		NDArray<T>* transpose(bool inplace = false) {
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