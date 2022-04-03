#pragma once


template <typename T>
class Matrix {
private:
	T* m_data;
	int m_width, m_height;

public:
	Matrix(T* data, int width, int height) {
		m_data = data;
		m_width = width;
		m_height = height;
	}

	Matrix(int width, int height) {
		m_data = (T*)malloc(width * height);
		m_width = width;
		m_height = height;
	}
};
