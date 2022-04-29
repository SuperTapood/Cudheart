#pragma once

#include "NDArray.cuh"
#include "../CUtil/CUtil.cuh"
#include "../Exceptions/Exceptions.cuh"

// check about using longs and stuff as lengths and indices for bigger tensors

namespace Cudheart::NDArrays {
	template <typename T>
	class Vector : public NDArray<T> {
	private:
		int m_size;
		T* m_data;

	public:
		Vector(T* data, int size) {
			m_data = data;
			m_size = size;
		}

		Vector(int size) {
			// m_data = (T*)malloc(size * sizeof(T));
			m_data = new T[size];
			m_size = size;
		}

		Vector(T* data) {
			m_data = data;
			m_size = (&data)[1] - data;
		}

		~Vector() {
			delete[] m_data;
		}

		template<typename U>
		Vector<U>* castTo() {
			Vector<U>* output = new Vector<U>(getSize());

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

		void set(int index, T value) {
			if (index < 0) {
				index += m_size;
			}
			m_data[index] = value;
		}

		int getSize() {
			return m_size;
		}

		string toString() {
			ostringstream os;
			os << "[";
			for (int i = 0; i < getSize() - 1; i++) {
				os << m_data[i] << ", ";
			}
			os << get(-1) << "]";
			return os.str();
		}

		void print() {
			cout << this->toString() << endl;
		}

		void printInfo() {
			cout << "Vector of size: " << m_size << endl;
		}
		// todo: add operator overloades to make this look better

		ContainerA<T>* getContainerA() {
			ContainerA<T>* out = new ContainerA<T>();

			out->warmUp((void**)m_data, m_size);

			return out;
		}

		ContainerAB<T>* getContainerAB(Vector<T>* other) {
			ContainerAB<T>* out = new ContainerAB<T>();

			out->warmUp((void**)m_data, (void**)other->m_data, m_size);

			return out;
		}

		ContainerABC<T>* getContainerABC(Vector<T>* b, Vector<T>* c) {
			ContainerABC<T>* out = new ContainerABC<T>();

			out->warmUp(m_data, b->m_data, c->m_data, m_size);

			return out;
		}

		void assertMatchSize(Vector<T>* other) {
			if (m_size != other->m_size) {
				Cudheart::Exceptions::ShapeMismatchException(m_size, other->m_size).raise();
			}
		}
	};
}