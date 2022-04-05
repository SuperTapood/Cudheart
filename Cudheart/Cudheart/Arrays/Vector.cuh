#pragma once

// check about using longs and stuff as lengths and indices for bigger tensors

template <typename T>
class Vector {
private:
	int m_size;
	T* m_data;

public:
	Vector(T* data, int size) {
		m_data = data;
		m_size = size;
	}

	Vector(int size) {
		m_data = (T*)malloc(size * sizeof(T));
		m_size = size;
	}

	Vector(T* data) {
		m_data = data;
		m_size = (&data)[1] - data;
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
	// todo: add operator overloades to make this look better
};