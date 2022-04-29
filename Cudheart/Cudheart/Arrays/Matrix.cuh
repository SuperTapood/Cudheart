#pragma once

// check about using longs and stuff as lengths and indices for bigger tensors

#include "../Util.cuh"
#include "../Exceptions/Exceptions.cuh"

namespace Cudheart::NDArrays {
	using namespace Cudheart::Exceptions;
	/// <summary>
	/// the matrix class
	/// </summary>
	/// <typeparam name="T">a provided type for the data contained in the class</typeparam>
	template <typename T>
	class Matrix {
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

		/// destroy the matrix object
		~Matrix() {
			delete[] m_data;
		}

		/// <summary>
		/// cast this matrix to a matrix of U type
		/// </summary>
		/// <typeparam name="U"> - the new type of the matrix</typeparam>
		/// <returns></returns>
		template<typename U>
		Matrix<U>* castTo() {
			Matrix<U>* output = new Matrix<U>(getSize());

			for (int i = 0; i < getSize(); i++) {
				output->set(i, (U)get(i));
			}

			return output;
		}

		/// <summary>
		/// get a value using a one dimensional index
		/// </summary>
		/// <param name="index"> - the index of the value in the actual raw data</param>
		/// <returns>the value at position i</returns>
		T get(int index) {
			if (index < 0) {
				index += m_size;
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
			return get(flatten(i, j));
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
			m_data[flatten(i, j)] = value;
		}

		/// <summary>
		/// get the element size of the matrix
		/// </summary>
		/// <returns>the element size</returns>
		int getSize() {
			return m_size;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns>the width of the matrix</returns>
		int getWidth() {
			return m_width;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns>the height of the matrix</returns>
		int getHeight() {
			return m_height;
		}

		/// <summary>
		/// convert the matrix to a string representation
		/// </summary>
		/// <returns>a string presenting the matrix</returns>
		string toString() {
 			ostringstream os;
			os << "[\n";
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
		Matrix<T>* dupe() {
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
		Matrix<T>* transpose(bool inplace) {
			Matrix<T>* mat = transpose();

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
		Matrix<T>* transpose() {
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
			Matrix<T>* mat = dupe();
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
		void assertMatchSize(Matrix<T>* other) {
			if (m_width != other->m_width || m_height != other->m_height) {
				Cudheart::Exceptions::ShapeMismatchException(m_width, m_height, other->m_width, other->m_height).raise();
			}
		}

		/// <summary>
		/// return a flattened vector with the same raw data as this matrix
		/// </summary>
		/// <returns>the output vector</returns>
		Vector<T>* flatten() {
			return VectorOps::fromMatrix(this);
		}

		// todo: add operator overloades to make this look better

	private:
		/// <summary>
		/// flatten a 2 dimensional index into a 1 dimensional index
		/// </summary>
		/// <param name="i"></param>
		/// <param name="j"></param>
		/// <returns>a flattened index</returns>
		int flatten(int i, int j) {
			return j + (i * m_width);
		}
	};
}