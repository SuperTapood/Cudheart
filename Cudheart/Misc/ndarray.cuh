#pragma once
#include <stdio.h>
#include <initializer_list>
#include <iostream>
#include <vector>
#include <type_traits>
#define FMT_HEADER_ONLY
#include <fmt/ranges.h>
#include <memory>

#include "promotion.cuh"

class Shape {
public:
	explicit Shape(std::vector<int> const& shape) {
		for (auto i : shape) {
			m_shape.push_back(i);
			m_size *= i;
		}
	}

	std::string toString() const {
		return fmt::format("({})", fmt::join(m_shape, ","));
	}

	long size() const {
		return m_size;
	}

	int ndims() const {
		return m_shape.size();
	}

	int at(int index) {
		return m_shape.at(index);
	}

	long subsize(int axis) {
		long out = 1;

		for (int i = axis + 1; i < ndims(); i++) {
			out *= m_shape.at(i);
		}

		return out;
	}

	Shape* copy() {
		return new Shape(m_shape);
	}

	std::vector<int> subshape(int axis) {
		std::vector<int> out;

		for (int i = 0; i < ndims(); i++) {
			if (i == axis) {
				continue;
			}
			out.push_back(m_shape.at(i));
		}

		return out;
	}

private:
	std::vector<int> m_shape;
	long m_size = 1;
};

#include <new>

template <typename T>
class NDArray {
public:
	T* m_data;
	Shape* m_shape;

public:
	NDArray(std::vector<int> const& shape) {
		m_shape = new Shape(shape);
		m_data = new T[size()];
	}

	NDArray(Shape* shape) {
		std::vector<int> s;

		for (int i = 0; i < shape->ndims(); i++) {
			s.push_back(shape->at(i));
		}

		m_shape = new Shape(s);
		m_data = new T[size()];
	}

	~NDArray() {
		delete[] m_data;
		delete m_shape;
	}

	long size() {
		return m_shape->size();
	}

	std::string toString() {
		std::string result = "[";
		for (int i = 0; i < size(); i++) {
			result += fmt::format("{}", m_data[i]);
			if (i != size() - 1) {
				result += ", ";
			}
		}
		result += "], shape = " + m_shape->toString();

		return result;
	}

	void println() {
		fmt::println(toString());
	}

	int ndims() const {
		return m_shape->ndims();
	}

	NDArray<T>* reshape(Shape* newShape, bool self = false) {
		if (newShape->size() != size()) {
			fmt::println("shape of size {} ({}) does not match shape of size {} ({})", newShape->size(), newShape->toString(), size(), m_shape->toString());
			exit(-1);
		}

		std::vector<int> shape;

		for (int i = 0; i < newShape->ndims(); i++) {
			shape.push_back(newShape->at(i));
		}

		if (self) {
			delete m_shape;

			m_shape = new Shape(shape);

			return this;
		}
		else {
			return copy()->reshape(shape, true);
		}
	}

	NDArray<T>* reshape(std::vector<int> const& newShape, bool self = false) {
		auto shape = new Shape(newShape);
		return reshape(shape, self);
	}

	long flattenIndex(std::vector<int> index) {
		if (index.size() != ndims()) {
			fmt::println("multi dimensional index of length {} does not match shape of {} dims", index.size(), ndims());
			exit(-1);
		}

		long flatIndex = 0;

		for (int i = 0; i < ndims(); i++) {
			auto subsize = m_shape->subsize(i);
			flatIndex += subsize * index.at(i);
		}

		return flatIndex;
	}

	void increment(std::vector<int>& indices, std::vector<int>& limits) const {
		for (int i = 0; i < indices.size(); i++) {
			indices.at(i)++;

			if (indices.at(i) == limits.at(i)) {
				indices.at(i) = 0;
			}
			else {
				return;
			}
		}
	}

	std::vector<int> getAxis(int axis, int index) {
		std::vector<int> indices;
		indices.reserve(ndims() - 1);

		for (int i = 0; i < ndims() - 1; i++) {
			indices.push_back(0);
		}

		std::vector<int> limits;
		limits.reserve(ndims() - 1);

		for (int i = 0; i < ndims(); i++) {
			if (i == axis) {
				continue;
			}
			limits.push_back(m_shape->at(i));
		}

		std::vector<int> axisIndices;

		long flat = index;

		int axisIndex = 0;

		/*do {

			auto temp = indices;

			temp.insert(temp.begin() + axis, index);

			flat = flattenIndex(temp);

			counter++;
			axisIndices.push_back(flat);

			increment(indices, limits);

			fmt::println("{}: ({})", fmt::join(temp, ","), flat);

		} while (counter < size() / m_shape->at(axis));

		return axisIndices;*/

		do {
			auto temp = indices;

			temp.insert(temp.begin() + axis, axisIndex++);

			flat = flattenIndex(temp) + index;

			axisIndices.push_back(flat);
		} while (axisIndex < m_shape->at(axis));

		return axisIndices;
	}

	T at(int index) {
		return m_data[index];
	}

	NDArray<T>* stretch(Shape* newShape) {
		if (newShape->ndims() != ndims()) {
			fmt::println("cannot stretch shape {} to shape {}", m_shape->toString(), newShape->toString());
			exit(-1);
		}
		NDArray<T>* result = new NDArray<T>(newShape);

		bool hasStretched = false;

		for (int axis = 0; axis < ndims(); axis++) {
			if (m_shape->at(axis) != 1 || newShape->at(axis) == 1) {
				continue;
			}

			hasStretched = true;
			for (int i = 0; i < m_shape->subsize(axis); i++) {
				auto indices = result->getAxis(axis, i);
				auto value = at(i);

				for (auto index : indices) {
					result->m_data[index] = value;
				}
			}
		}

		return hasStretched ? result : this;
	}

	NDArray<T>* stretch(std::vector<int> const& newShape) {
		auto shape = new Shape(newShape);
		return stretch(shape);
	}

	NDArray<T>* copy() {
		auto out = new NDArray<T>(m_shape->copy());
		std::copy(m_data, m_data + size(), out->m_data);
		return out;
	}

	NDArray<T>* subarray(std::vector<int> const& indices) {
		auto out = new NDArray<T>({ (int)indices.size() });

		for (int i = 0; i < indices.size(); i++) {
			out->m_data[i] = m_data[indices.at(i)];
		}

		return out;
	}
};


template <typename A, typename B>
std::pair<NDArray<A>*, NDArray<B>*> broadcast(NDArray<A>* a, NDArray<B>* b) {
	a = a->copy();
	b = b->copy();

	int diff = a->ndims() - b->ndims();
	std::vector<int> newShape;

	for (int i = 0; i < std::abs(diff); i++) {
		newShape.push_back(1);
	}
	
	if (a->ndims() > b->ndims()) {
		for (int i = 0; i < b->ndims(); i++) {
			newShape.push_back(b->m_shape->at(i));
		}
		b->reshape(newShape, true);
	} else if (b->ndims() > a->ndims()) {
		for (int i = 0; i < a->ndims(); i++) {
			newShape.push_back(a->m_shape->at(i));
		}
		a->reshape(newShape, true);
	}

	newShape.clear();

	for (int i = 0; i < a->ndims(); i++) {
		auto sa = a->m_shape->at(i);
		auto sb = b->m_shape->at(i);

		if (sa == sb) {
			newShape.push_back(sa);
			continue;
		}

		if (sa == 1) {
			newShape.push_back(sb);
			continue;
		}
		else if (sb == 1) {
			newShape.push_back(sa);
			continue;
		}

		fmt::println("shapes {} and {} cannot be broadcasted", a->m_shape->toString(), b->m_shape->toString());
		exit(-1);
	}

	//fmt::println("final shape: ({})", fmt::join(newShape, ","));

	a = a->stretch(newShape);
	b = b->stretch(newShape);

	return {a, b};
}

template <typename A, typename B, typename T = promote2(A, B)>
NDArray<T>* add(NDArray<A>* x, NDArray<B>* y) {
	auto [a, b] = broadcast(x, y);

	auto result = new NDArray<T>(a->m_shape);

#pragma omp parallel for
	for (int i = 0; i < result->size(); i++) {
		result->m_data[i] = (T)a->at(i) + (T)b->at(i);
	}

	return result;
}

template <typename A>
A sum(NDArray<A>* x) {
	A total = 0;

	for (int i = 0; i < x->size(); i++) {
		total += x->at(i);
	}

	return total;
}

template <typename A>
NDArray<A>* sum(NDArray<A>* x, int axis) {
	auto result = new NDArray<A>(x->m_shape->subshape(axis));

	int index = 0;

	for (int idx = 0; idx < x->m_shape->subsize(axis); idx++) {
		auto indices = x->getAxis(axis, idx);
		result->m_data[index++] = sum(x->subarray(indices));
	}

	return result;
}


void test() {
	auto a = new NDArray<int>({ 5, 5 });
	auto b = new NDArray<float>({ 3, 5, 5 });

	for (int i = 0; i < a->size(); i++) {
		a->m_data[i] = i;
	}

	for (int i = 0; i < b->size(); i++) {
		b->m_data[i] = (float)i;
	}

	//b->println();
	//fmt::println("{}", b->getAxis(0, 0));

	//a->stretch(b->m_shape)->println();

	/*auto [ra, rb] = broadcast(a, b);

	ra->println();
	rb->println();
	fmt::println("a - {}\nb - {}", a->m_shape->toString(), b->m_shape->toString());*/

	auto r = add(a, b);

	fmt::println("{}", sum(a));

	sum(b, 0)->println();

	/*delete r;
	delete a;
	delete b;
	delete ra;
	delete rb;*/
}