#pragma once

#include "NDArray.cuh"

// check about using longs and stuff as lengths and indices for bigger tensors

namespace Cudheart::NDArrays {
	template <typename T>
	class Matrix : public NDArray<T> {
	private:
		int m_width, m_height, m_size;
		T* m_data;

	public:
		Matrix(T* data, int width, int height) {
			m_data = data;
			m_width = width;
			m_height = height;
			m_size = width * height;
		}

		Matrix(int width, int height) {
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

		virtual T get(int index) {
			if (index < 0) {
				index += m_size;
			}
			return m_data[index];
		}

		T get(int i, int j) {

			if (i < 0) {
				i += m_width;
			}
			if (j < 0) {
				j += m_height;
			}
			int index = flatten(i, j);
			return m_data[index];
		}

		virtual void set(int index, T value) {
			if (index < 0) {
				index += m_size;
			}
			m_data[index] = value;
		}

		void set(int i, int j, T value) {
			if (i < 0) {
				i += m_width;
			}
			if (j < 0) {
				j += m_height;
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
			for (int i = 0; i < m_width; i++) {
				os << " [";
				for (int j = 0; j < m_height; j++) {
					if (j + 1 == m_height) {
						os << get(i, j);
					}
					else {
						os << get(i, j) << ", ";
					}
				}
				if (i + 1 == m_width) {
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
			cout << "Matrix of size: " << m_size << " width " << m_width << " and height " << m_height << endl;
		}

		Matrix<T>* dupe() {
			Matrix<T>* out = new Matrix<T>(m_width, m_height);
			
			for (int i = 0; i < m_size; i++) {
				out->set(i, get(i));
			}

			return out;
		}
		// todo: add operator overloades to make this look better

		int flatten(int i, int j) {
			return j + (i * m_height);
		}
	};
}