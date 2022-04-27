#pragma once

// check about using longs and stuff as lengths and indices for bigger tensors

#include "../Util.cuh"

namespace Cudheart::NDArrays {
	template <typename T>
	class Matrix {
	private:
		int m_width, m_height, m_size;
		T* m_data;

	public:
		Matrix(T* data, int height, int width) {
			m_data = data;
			m_width = width;
			m_height = height;
			m_size = width * height;
		}

		Matrix(int height, int width) {
			// m_data = (T*)malloc(static_cast<unsigned long long>(width) * height * sizeof(T));
			m_data = new T[width * height];
			m_width = width;
			m_height = height;
			m_size = width * height;
		}

		~Matrix() {
			delete[] m_data;
		}

		template<typename U>
		Matrix<U>* castTo() {
			Matrix<U>* output = new Matrix<U>(getSize());

			for (int i = 0; i < getSize(); i++) {
				output->set(i, (U)get(i));
			}

			return output;
		}

		T get(int index) {
			if (index < 0) {
				index += m_size;
			}
			return m_data[index];
		}

		T get(int i, int j) {

			if (i < 0) {
				i += m_height;
			}
			if (j < 0) {
				j += m_width;
			}
			return get(flatten(i, j));
		}

		void set(int index, T value) {
			if (index < 0) {
				index += m_size;
			}
			m_data[index] = value;
		}

		void set(int i, int j, T value) {
			if (i < 0) {
				i += m_height;
			}
			if (j < 0) {
				j += m_width;
			}
			m_data[flatten(i, j)] = value;
		}

		int getSize() {
			return m_size;
		}

		int getWidth() {
			return m_width;
		}

		int getHeight() {
			return m_height;
		}

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

		void print() {
			cout << this->toString() << endl;
		}

		void printInfo() {
			cout << "Matrix of size: " << to_string(m_size) << " width " << to_string(m_width) << " and height " << to_string(m_height) << endl;
		}

		Matrix<T>* dupe() {
			Matrix<T>* out = new Matrix<T>(m_height, m_width);

			for (int i = 0; i < m_size; i++) {
				out->set(i, get(i));
			}

			return out;
		}

		Matrix<T>* transpose() {
			Matrix<T>* mat = new Matrix<T>(m_width, m_height);

			for (int i = 0; i < m_width; i++) {
				for (int j = 0; j < m_height; j++) {
					mat->set(i, j, get(j, i));
				}
			}

			return mat;

		}

		Matrix<T>* reverseRows() {
			Matrix<T>* mat = new Matrix<T>(m_height, m_width);


			for (int i = 0; i < m_height; i++) {
				for (int j = m_width - 1, k = 0; j > -1; j--, k++) {
					mat->set(i, k, get(i, j));
				}
			}

			return mat;
		}

		// angles = dir * 90
		Matrix<T>* rotate(int angles) {
			Matrix<T>* mat = dupe();
			Matrix<T>* out = nullptr;

			if (angles == 90) {
				Matrix<T>* trans = mat->transpose();
				delete mat;

				Matrix<T>* rev = trans->reverseRows();
				delete trans;

				out = rev;
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
				Matrix<T>* rev = mat->reverseRows();
				delete mat;

				Matrix<T>* trans = rev->transpose();
				delete rev;

				out = trans;
			}
			
			return out;
		}

		Matrix<T>* flip() {
			return rotate(180);
		}

		// todo: add operator overloades to make this look better

	private:
		int flatten(int i, int j) {
			return j + (i * m_width);
		}
	};
}