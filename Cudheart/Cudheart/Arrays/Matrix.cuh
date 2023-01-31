#pragma once

// check about using longs and stuff as lengths and indices for bigger tensors

#include "../Util.cuh"
#include "../Exceptions/Exceptions.cuh"
#include "Vector.cuh"
#include "Shape.cuh"

namespace Cudheart {
	namespace NDArrays {
		using namespace Cudheart::Exceptions;
		/// <summary>
		/// the matrix class
		/// </summary>
		/// <typeparam name="T"> - provided type for the data contained in the class</typeparam>
		template <typename T>
		class Matrix : public NDArray<T> {
		private:
			/// <summary>
			/// the width of the matrix
			/// </summary>
			int m_width;
			/// <summary>
			/// the height of the matrix
			/// </summary>
			int m_height;
			/// <summary>
			/// the overall element count of the matrix
			/// </summary>
			int m_size;

			/// <summary>
			/// the raw data of the matrix
			/// </summary>
			T* m_data;

		public:

#pragma region constructors
			/// <summary>
			/// create a new matrix object from existing data
			/// </summary>
			/// <param name="data"> - raw data to be provided</param>
			/// <param name="height"> - the height of the matrix</param>
			/// <param name="width"> - the width of the matrix</param>
			Matrix(T* data, int height, int width) {
				m_data = data;
				m_width = width;
				m_height = height;
				m_size = width * height;
			}

			/// <summary>
			/// create a new empty matrix from garbage values
			/// </summary>
			/// <param name="height"> - the height of the matrix</param>
			/// <param name="width"> - the width of the matrix</param>
			Matrix(int height, int width) {
				m_data = new T[width * height];
				m_width = width;
				m_height = height;
				m_size = width * height;
			}

			/// <summary>
			/// create a matrix out of a initilizer list
			/// </summary>
			/// <param name="list"> - the given init list</param>
			/// <param name="height"> - the height of the matrix</param>
			/// <param name="width"> - the width of the matrix</param>
			Matrix(initializer_list<T> list, int height, int width) {
				m_size = list.size();
				m_width = width;
				m_height = height;
				// assert that the width and height are of the correct size
				if (m_size != width * height) {
					cout << width * height << endl;
					cout << list.size() << endl;
					BadValueException("Matrix creator ", "width: " + to_string(width) + " and height : " + to_string(height), to_string(m_size) + " (same as amount of initializer list elements provided)");
				}
				m_data = new T[m_size];
				int i = 0;
				// iterate through the list
				for (auto& x : list) {
					m_data[i] = x;
					i++;
				}
			}

			Matrix(Shape* shape) {
				int height = shape->getX();
				int width = shape->getY();
				m_data = new T[width * height];
				m_width = width;
				m_height = height;
				m_size = width * height;
			}
#pragma endregion

			/// destroy the matrix object
			~Matrix() {
				// delete the raw data
				delete[] m_data;
			}

			/// <summary>
			/// cast this matrix to a matrix of U type
			/// </summary>
			/// <typeparam name="U"> - the new type of the matrix</typeparam>
			/// <returns></returns>
			template<typename U>
			Matrix<U>* castTo() {
				if constexpr (std::is_same_v<T, U>) {
					return this;
				}
				if constexpr (std::is_same_v<U, StringType*>) {
					// Convert the vector of T to a vector of StringType*.
					// For example, if T is a numeric type, then each element of the vector
					// will be converted to a string representation.
					Matrix<StringType*>* out = new Matrix<StringType*>(getHeight(), getWidth());
					for (int i = 0; i < getSize(); i++) {
						out->set(i, new StringType(getString(i)));
					}
					return (Matrix<U>*)out;
				}
				else if constexpr (std::is_same_v<U, ComplexType*>) {
					if constexpr (std::is_arithmetic_v<T>) {
						// Convert the vector of T to a vector of ComplexType*.
						// For example, if T is a numeric type, then each element of the vector
						// will be wrapped in a ComplexType object.
						Matrix<ComplexType*>* out = new Matrix<ComplexType*>(getHeight(), getWidth());
						for (int i = 0; i < getSize(); i++) {
							out->set(i, new ComplexType(get(i)));
						}
						return (Matrix<U>*)out;
					}
					else if constexpr (std::is_same_v<T, StringType*>) {
						// Convert the vector of T to a vector of ComplexType*.
						// For example, if T is a numeric type, then each element of the vector
						// will be wrapped in a ComplexType object.
						Matrix<ComplexType*>* out = new Matrix<ComplexType*>(getHeight(), getWidth());
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
						return (Matrix<U>*)out;
					}
				}
				else if constexpr (std::is_arithmetic_v<U>) {
					if constexpr (std::is_same_v<T, StringType*>) {
						// Convert the vector of StringType* to a vector of U.
						// For example, if U is a numeric type, then each element of the vector
						// will be converted from a string representation to a numeric value.
						Matrix<U>* out = new Matrix<U>(getHeight(), getWidth());
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
						Matrix<U>* out = new Matrix<U>(getHeight(), getWidth());
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

#pragma region getters_and_setters

			/// <summary>
			/// get a value using a one dimensional index
			/// </summary>
			/// <param name="index"> - the index of the value in the actual raw data</param>
			/// <returns>the value at position i</returns>
			T get(int index) {
				if (index < 0) {
					index += m_size;
				}

				if (index > m_size) {
					IndexOutOfBoundsException(m_width, m_height, index);
				}
				return m_data[index];
			}

			/// <summary>
			/// get a value using a two dimensional index
			/// </summary>
			/// <param name="i"> - col, i think</param>
			/// <param name="j"> - pretty sure this means row</param>
			/// <returns>the value at position j + (i * m_width)</returns>
			T get(int i, int j) {
				if (i < 0) {
					i += m_height;
				}
				if (j < 0) {
					j += m_width;
				}
				if (i >= m_height || j >= m_width) {
					IndexOutOfBoundsException(m_width, m_height, i, j);
				}
				return get(flatten(i, j));
			}

			string getString(int index) {
				if constexpr (is_same_v<T, StringType*>) {
					return ((StringType*)(get(index)))->toString();
				}
				else if constexpr (is_same_v<T, ComplexType*>) {
					return ((ComplexType*)(get(index)))->toString();
				}
				else if constexpr (is_same_v<T, NDArray<T>*>) {
					return ((NDArray<T>*)(get(index)))->toString();
				}
				else {
					return to_string(get(index));
				}
				return "";
			}

			string getString(int i, int j) {
				return getString(flatten(i, j));
			}

			template <typename U>
			NDArray<T>* emptyLike() {
				return new Matrix<U>(m_height, m_width);
			}

			NDArray<T>* emptyLike() {
				return new Matrix<T>(m_height, m_width);
			}

			template <typename U>
			NDArray<U>* reshape(Shape* shape) {
				if (shape->getX() * shape->getY() != m_size) {
					ostringstream os;
					os << "(" << m_height << ", " << m_width << ")";
					throw ShapeMismatchException(os.str(), shape->toString());
				}

				if (shape->getDims() == 2) {
					auto out = new Matrix<U>(shape);

					for (int i = 0; i < m_size; i++) {
						out->set(i, (U)get(i));
					}

					return out;
				}

				return castTo<U>()->flatten();
			}

			int getDims() const {
				return 2;
			}

			/// <summary>
			/// fetch a specific row out of the matrix
			/// </summary>
			/// <param name="i"> - the index of the row</param>
			/// <returns>the row as vector</returns>
			Vector<T>* getRow(int i) {
				if (i >= m_height) {
					IndexOutOfBoundsException(m_width, m_height, i);
				}
				Vector<T>* out = new Vector<T>(m_width);
				for (int k = 0; k < m_width; k++) {
					out->set(k, get(i, k));
				}
				return out;
			}

			/// <summary>
			/// set value at position i to value
			/// </summary>
			/// <param name="index"> - the index of the value</param>
			/// <param name="value"> - the value to set to</param>
			void set(int index, T value) {
				if (index < 0) {
					index += m_size;
				}
				if (index > m_size) {
					IndexOutOfBoundsException(m_width, m_height, index);
				}
				m_data[index] = value;
			}

			/// <summary>
			/// set value at position j + (i * m_width) to value
			/// </summary>
			/// <param name="i"> - col?</param>
			/// <param name="j"> - row?</param>
			/// <param name="value"> - the value to set to</param>
			void set(int i, int j, T value) {
				if (i < 0) {
					i += m_height;
				}
				if (j < 0) {
					j += m_width;
				}
				if (i >= m_height || j >= m_width) {
					IndexOutOfBoundsException(m_width, m_height, i, j);
				}
				set(flatten(i, j), value);
			}

			/// <summary>
			/// get the element size of the matrix
			/// </summary>
			/// <returns>the element size</returns>
			int getSize() const {
				return m_size;
			}

			/// <summary>
			/// get the width of the matrix
			/// </summary>
			/// <returns>the width of the matrix</returns>
			int getWidth() {
				return m_width;
			}

			/// <summary>
			/// get the height of the matrix
			/// </summary>
			/// <returns>the height of the matrix</returns>
			int getHeight() {
				return m_height;
			}

#pragma endregion

			/// <summary>
			/// convert the matrix to a string representation
			/// </summary>
			/// <returns>a string presenting the matrix</returns>
			string toString() {
				ostringstream os;
				os << "[\n";
				if constexpr (is_same_v<T, StringType*>) {
					for (int i = 0; i < m_height; i++) {
						os << " [";
						for (int j = 0; j < m_width; j++) {
							if (j + 1 == m_width) {
								os << ((StringType*)get(i, j))->toString();
							}
							else {
								T res = get(i, j);
								os << ((StringType*)res)->toString() << ", ";
							}
						}
						if (i + 1 == m_height) {
							os << "]\n";
						}
						else {
							os << "],\n";
						}
					}
				}
				else if constexpr (is_same_v<T, ComplexType*>) {
					for (int i = 0; i < m_height; i++) {
						os << " [";
						for (int j = 0; j < m_width; j++) {
							if (j + 1 == m_width) {
								os << ((ComplexType*)get(i, j))->toString();
							}
							else {
								T res = get(i, j);
								os << ((ComplexType*)res)->toString() << ", ";
							}
						}
						if (i + 1 == m_height) {
							os << "]\n";
						}
						else {
							os << "],\n";
						}
					}
				}
				else {
					for (int i = 0; i < m_height; i++) {
						os << " [";
						for (int j = 0; j < m_width; j++) {
							T res = get(i, j);
							os << std::setprecision(17) << res;
							if constexpr (std::is_floating_point_v<T>) {
								if ((long)res == res) {
									os << ".0";
								}
							}
							if (j + 1 != m_width) {
								os << ", ";
							}
						}
						if (i + 1 == m_height) {
							os << "]\n";
						}
						else {
							os << "],\n";
						}
					}
				}
				os << "]";
				return os.str();
			}

			/// <summary>
			/// print the matrix
			/// </summary>
			void print() {
				cout << this->toString() << endl;
			}

			/// <summary>
			/// print info about the matrix
			/// </summary>
			void printInfo() {
				cout << "Matrix of size: " << to_string(m_size) << " width " << to_string(m_width) << " and height " << to_string(m_height) << endl;
			}

			/// <summary>
			/// dupe the matrix
			/// </summary>
			/// <returns>a copy of this matrix</returns>
			NDArray<T>* copy() {
				Matrix<T>* out = new Matrix<T>(m_height, m_width);

				for (int i = 0; i < m_size; i++) {
					out->set(i, get(i));
				}

				return out;
			}

			/// <summary>
			/// transpose the matrix
			/// </summary>
			/// <param name="inplace"> - whether or not to transpose this matrix, or a copy of it</param>
			/// <returns>this matrix transposed if inplace is true, else a transposed copy</returns>
			NDArray<T>* transpose(bool inplace = false) {
				Matrix<T>* mat = new Matrix<T>(m_width, m_height);

				for (int i = 0; i < m_height; i++) {
					for (int j = 0; j < m_width; j++) {
						mat->set(j, i, get(i, j));
					}
				}

				if (inplace) {
					m_height = mat->getHeight();
					m_width = mat->getWidth();

					for (int i = 0; i < m_size; i++) {
						m_data[i] = mat->m_data[i];
					}

					delete mat;

					return this;
				}

				return mat;
			}

			/// <summary>
			/// reverse the rows of this matrix
			/// </summary>
			/// <param name="inplace"> - whether or not to reverse the rows of this matrix or of a copy of it</param>
			/// <returns>this matrix with reversed rows if inplace is true, else a copy with its rows reversed</returns>
			Matrix<T>* reverseRows(bool inplace = false) {
				Matrix<T>* mat = new Matrix<T>(m_height, m_width);

				for (int i = 0; i < m_height; i++) {
					for (int j = 0; j < m_width; j++) {
						mat->set(i, m_width - j - 1, get(i, j));
					}
				}

				if (inplace) {
					setData(mat);
					delete mat;
					return this;
				}

				return mat;
			}

			Matrix<T>* reverseCols(bool inplace = false) {
				Matrix<T>* mat = new Matrix<T>(m_height, m_width);

				for (int i = 0; i < m_height; i++) {
					for (int j = 0; j < m_width; j++) {
						mat->set(m_height - i - 1, j, get(i, j));
					}
				}

				if (inplace) {
					setData(mat);
					delete mat;
					return this;
				}

				return mat;
			}

			Matrix<T>* rot90(int k, bool inplace = false) {
				Matrix<T>* mat = (Matrix<T>*)copy();

				for (int i = 0; i < (k % 4); i++) {
					mat = ((Matrix<T>*)(mat->transpose()))->reverseCols();
				}

				if (inplace) {
					setData(mat);
					delete mat;
					return this;
				}

				return mat;
			}

			Matrix<T>* rot90(bool inplace = false) {
				return rot90(1, inplace);
			}

			/// <summary>
			/// flip this matrix
			/// </summary>
			/// <param name="inplace">whether or not to flip this matrix, or a copy of it</param>
			/// <returns>if inplace is true, this matrix, otherwise a flipped copy</returns>
			Matrix<T>* flip(bool inplace = false) {
				return rot90(2, inplace);
			}

			/// <summary>
			/// set the raw data of this matrix according to another
			/// </summary>
			/// <param name="mat">the matrix to set this one to</param>
			void setData(Matrix<T>* mat) {
				if (getWidth() != mat->getWidth() || getHeight() != mat->getHeight()) {
					m_data = new T[mat->getWidth() * mat->getHeight()];
				}

				for (int i = 0; i < m_size; i++) {
					set(i, mat->get(i));
				}
			}

			/// <summary>
			/// assert that the dims of this matrix equal the dims of matrix other. if they are not an exception is "raised"
			/// </summary>
			/// <param name="other">the matrix to compare to</param>
			void assertMatchShape(Shape* shape, int axis = 0) {
				if (shape->getDims() == 2) {
					if (getWidth() != shape->getX() || getHeight() != shape->getY()) {
						ShapeMismatchException(m_width, m_height, shape->getX(), shape->getY()).raise();
					}
				}
				else if (shape->getDims() == 1) {
					if (axis == 0) {
						if (m_width != shape->getSize()) {
							Cudheart::Exceptions::ShapeMismatchException(getShape()->toString(),
								shape->toString()).raise();
						}
					}
					else if (axis == 1) {
						if (m_height != shape->getSize()) {
							Cudheart::Exceptions::ShapeMismatchException(getShape()->toString(),
								shape->toString()).raise();
						}
					}
				}
			}

			/// <summary>
			/// return a flattened vector with the same raw data as this matrix
			/// </summary>
			/// <returns>the output vector</returns>
			Vector<T>* flatten() {
				Vector<T>* out = new Vector<T>(getSize());

				for (int i = 0; i < getSize(); i++) {
					out->set(i, get(i));
				}

				return out;
			}

			Shape* getShape() {
				return new Shape(m_width, m_height);
			}

			Matrix<T>* augment(Vector<T>* other) {
				assertMatchShape(other->getShape(), 1);

				Matrix<T>* res = new Matrix<T>(m_height, m_width + 1);

				for (int i = 0; i < m_height; i++) {
					for (int j = 0; j < m_width; j++) {
						res->set(i, j, get(i, j));
					}
				}

				for (int i = 0; i < m_height; i++) {
					res->set(i, -1, other->get(i));
				}

				return res;
			}

			// todo add operator overloads to make this look better and add some augment overloads

		private:
			/// <summary>
			/// flatten a 2 dimensional index into a 1 dimensional index
			/// </summary>
			/// <param name="i"> - index</param>
			/// <param name="j"> - jdex</param>
			/// <returns>a flattened index</returns>
			int flatten(int i, int j) {
				return j + (i * m_width);
			}
		};
	}
}