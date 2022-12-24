#pragma once

// check about using longs and stuff as lengths and indices for bigger tensors

#include "../Util.cuh"
#include "../Exceptions/Exceptions.cuh"
#include "Vector.cuh"
#include "Shape.cuh"

namespace Cudheart::NDArrays {
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
				BadValueException("Matrix creator ", "width: " + to_string(width) + " and height : " + to_string(height), to_string(width * height) + " (same as amount of initializer list elements provided)");
			}
			m_data = new T[m_size];
			int i = 0;
			// iterate through the list
			for (auto& x : list) {
				m_data[i] = x;
				i++;
			}
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
			constexpr bool isStringType = is_same_v<U, StringType*>;
			constexpr bool amStringType = is_same_v<T, StringType*>;
			constexpr bool isComplexType = is_same_v<U, ComplexType*>;
			constexpr bool amArithmetic = is_arithmetic_v<T>;
			constexpr bool isArithmetic = is_arithmetic_v<U>;
			constexpr bool isVoid = is_void_v<U>;
			constexpr bool isNull = is_null_pointer_v<U>;
			if (isStringType) {
				auto out = new Matrix<StringType*>(getHeight(), getWidth());

				for (int i = 0; i < getSize(); i++) {
					out->set(i, new StringType(getString(i)));
				}

				return (Matrix<U>*)out;
			}
			else if (isComplexType) {
				if (amArithmetic) {
					auto out = new Matrix<ComplexType*>(getHeight(), getWidth());

					for (int i = 0; i < getSize(); i++) {
						out->set(i, new ComplexType(get(i)));
					}

					return (Matrix<U>*)out;
				}
			}
			else if (isArithmetic) {
				if (amStringType) {
					auto out = new Matrix<U>(getHeight(), getWidth());

					for (int i = 0; i < getSize(); i++) {
						auto str = (StringType*)get(i);
						out->set(i, (U)str->toFloating());
					}

					return out;
				} else if (!isVoid) {
					if (!isNull) {
						if (amArithmetic) {
							auto out = new Matrix<U>(getHeight(), getWidth());

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
			else {
				ostringstream os;
				os << "BadTypeException: cannot cast matrix of type ";
				os << typeid(T).name();
				os << " to a matrix of type ";
				os << typeid(U).name();
				BadTypeException(os.str());
			}

			cout << "oh no" << endl;
			return NULL;
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
		NDArray<T>* reshape(Shape* other) {
			assertMatchShape(other);
			if (other->getDims() == 2) {
				return this;
			}
			else if (other->getDims() == 1) {
				auto out = new Vector<T>(other->getSize());

				for (int i = 0; i < m_size; i++) {
					out->set(i, get(i));
				}

				return v;
			}

			return nullptr;
		}

		int getDims() {
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
		int getSize() {
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
						if (j + 1 == m_width) {
							os << get(i, j);
						}
						else {
							T res = get(i, j);
							os << res << ", ";
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
		NDArray<T>* transpose(bool inplace) {
			Matrix<T>* mat = (Matrix<T>*)transpose();

			if (inplace) {
				setData(mat);
				delete mat;
				return this;
			}

			return mat;
		}

		/// <summary>
		/// transpose a copy of this matrix
		/// </summary>
		/// <returns>a transposed copy of this matrix</returns>
		NDArray<T>* transpose() {
			Matrix<T>* mat = new Matrix<T>(m_width, m_height);

			for (int i = 0; i < m_width; i++) {
				for (int j = 0; j < m_height; j++) {
					mat->set(i, j, get(j, i));
				}
			}

			return mat;
		}

		/// <summary>
		/// reverse the rows of this matrix
		/// </summary>
		/// <param name="inplace"> - whehter or not to reverse the rows of this matrix or of a copy of it</param>
		/// <returns>this matrix with reversed rows if inplace is true, else a copy with its rows reversed</returns>
		Matrix<T>* reverseRows(bool inplace) {
			Matrix<T>* mat = reverseRows();

			if (inplace) {
				setData(mat);
				delete mat;
				return this;
			}

			return mat;
		}

		/// <summary>
		/// reverse the rows of this matrix
		/// </summary>
		/// <returns>a copy of this matrix with its rows reversed</returns>
		Matrix<T>* reverseRows() {
			Matrix<T>* mat = new Matrix<T>(m_height, m_width);

			for (int i = 0; i < m_height; i++) {
				for (int j = m_width - 1, k = 0; j > -1; j--, k++) {
					mat->set(i, k, get(i, j));
				}
			}

			return mat;
		}

		/// <summary>
		/// rotate this matrix
		/// </summary>
		/// <param name="degrees"> - the amount of degrees to rotate this matrix for (degrees % 90 == 0)</param>
		/// <param name="inplace"> - whether to rotate this matrix, or a copy of it</param>
		/// <returns></returns>
		Matrix<T>* rotate(int degrees, bool inplace) {
			Matrix<T>* mat = rotate(degrees);

			if (inplace) {
				setData(mat);
				delete mat;
				return this;
			}

			return mat;
		}

		/// <summary>
		/// rotate this matrix
		/// </summary>
		/// <param name="angles"> - degrees to rotate by</param>
		/// <returns>a rotated copy of this matrix</returns>
		Matrix<T>* rotate(int angles) {
			Matrix<T>* mat = (Matrix<T>*)copy();
			Matrix<T>* out = nullptr;

			if (angles == 90) {
				mat->transpose(true);
				mat->reverseRows(true);
				out = mat;
			}
			else if (angles == 180) {
				out = mat->rotate(90)->rotate(90);
			}
			else if (angles == 270) {
				out = rotate(-90);
			}
			else if (angles == -180) {
				out = rotate(180);
			}
			else if (angles == -90) {
				mat->reverseRows(true);
				mat->transpose(true);
				out = mat;
			}

			return out;
		}

		/// <summary>
		/// flip this matrix
		/// </summary>
		/// <param name="inplace">whether or not to flip this matrix, or a copy of it</param>
		/// <returns>if inplace is true, this matrix, otherwise a flipped copy</returns>
		Matrix<T>* flip(bool inplace) {
			return rotate(180, inplace);
		}

		/// <summary>
		/// flip this matrix
		/// </summary>
		/// <returns>a flipped copy</returns>
		Matrix<T>* flip() {
			return rotate(180);
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
		void assertMatchShape(Shape* shape, int axis) {
			if (shape->getDims() == 2) {
				if (getWidth() != shape->getX() || getHeight() != shape->getY()) {
					ShapeMismatchException(m_width, m_height, shape->getX(), shape->getY()).raise();
				}
			}
			else if (shape->getDims() == 1) {
				if (axis == 0) {
					if (m_width != shape->getSize()) {
						Cudheart::Exceptions::ShapeMismatchException(m_width,
							shape->getSize()).raise();
					}
				}
				else if (axis == 1) {
					if (m_height != shape->getSize()) {
						Cudheart::Exceptions::ShapeMismatchException(m_height,
							shape->getSize()).raise();
					}
				}
			}
		}

		void assertMatchShape(Shape* shape) {
			return assertMatchShape(shape, 0);
		}

		/// <summary>
		/// return a flattened vector with the same raw data as this matrix
		/// </summary>
		/// <returns>the output vector</returns>
		NDArray<T>* flatten() {
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

			Matrix<T>* res = new Matrix<T>(m_width + 1, m_height);

			for (int i = 0; i < m_height; i++) {
				for (int j = 0; j < m_width; j++) {
					res->set(i, j, get(i, j));
				}
				res->set(i, m_width, other->get(i));
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