#pragma once

#include <vector>
#include <sstream>
#include <algorithm>

#define FMT_HEADER_ONLY
#include "fmt/format.h"

#include "NDArrayIters.cuh"
#include "NDArrayBase.cuh"

namespace CudheartNew {
	template <typename T>
	class NDArray : public NDArrayBase {
	private:
		T* m_data;

	public:
		explicit NDArray(Shape shape) {
			m_shape = shape;
			for (auto dim : shape) {
				m_size *= dim;
			}
			m_data = new T[m_size];
		}

		explicit NDArray(long size) {
			m_shape = { size };

			m_size = size;

			m_data = new T[m_size];
		}

		~NDArray() {
			delete[] m_data;
		}

		std::string printRecursive(long* s, int len, int start, int offset) final
		{
			std::ostringstream os;
			os << "[";
			if (len == start + 1) {
				for (int i = 0; i < s[start]; i++) {
					if (i != 0) {
						os << " ";
					}

					os << m_data[offset + i];

					if (i != s[start] - 1) {
						os << ",";
					}
				}
			}
			else {
				os << "\n";
				for (int i = 0; i < s[start]; i++) {
					for (int i = 0; i <= start; i++) {
						os << "  ";
					}
					os << printRecursive(s, len, start + 1, offset) << ",\n";
					offset += subsize(start);
				}
				for (int i = 0; i <= start; i++) {
					os << " ";
				}
			}
			os << "]";
			return os.str();
		}

		NDArray<T>* reshape(Shape newShape, bool self = false) {
			int newsize = 1;
			for (auto val : newShape) {
				newsize *= val;
			}
			if (newsize != size()) {
				println();
				fmt::println("shape of size {} ({}) does not match shape of size {} ({})", newsize, fmt::join(newShape, ","), size(), shapeString());
				exit(-1);
			}

			if (self) {
				m_shape = newShape;
				return this;
			}
			else {
				return copy()->reshape(newShape, true);
			}
		}

		long flattenIndex(std::vector<long> const& index) {
			if (index.size() != ndims()) {
				fmt::println("multi dimensional index of length {} does not match shape of {} dims", index.size(), ndims());
				exit(-1);
			}

			long flatIndex = 0;

			for (int i = 0; i < ndims(); i++) {
				auto sub = subsize(i);
				flatIndex += sub * index.at(i);
			}

			return flatIndex;
		}

		void increment(Shape& indices, std::vector<long>& limits) const {
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

		std::vector<long> getAxis(int axis, int index) {
			if (axis < 0) {
				fmt::println("axis cannot be less than 0");
				exit(-1);
			}
			std::vector<long> indices;
			indices.reserve(ndims() - 1);

			for (int i = 0; i < ndims() - 1; i++) {
				indices.push_back(0);
			}

			std::vector<long> limits;
			limits.reserve(ndims() - 1);

			for (int i = 0; i < ndims(); i++) {
				if (i == axis) {
					continue;
				}
				limits.push_back(m_shape.at(i));
			}

			std::vector<long> axisIndices;

			long flat = index;

			int axisIndex = 0;

			do {
				auto temp = indices;

				temp.insert(temp.begin() + axis, axisIndex++);

				flat = flattenIndex(temp) + index;

				axisIndices.push_back(flat);
			} while (axisIndex < m_shape.at(axis));

			return axisIndices;
		}

		T& at(int index) {
			if (index < 0) {
				index = size() + index;
			}
			return m_data[index];
		}

		const T& at(int index) const {
			if (index < 0) {
				index = size() + index;
			}
			return m_data[index];
		}

		T& at(std::vector<long>& indices) {
			return at(flattenIndex(indices));
		}

		const T& at(std::vector<long>& indices) const {
			return at(flattenIndex(indices));
		}

		NDArray<T>* stretch(Shape const& newShape) {
			if (newShape.size() != ndims()) {
				fmt::println("cannot stretch shape {} to shape {}", shapeString(), fmt::join(newShape, ","));
				exit(-1);
			}
			NDArray<T>* result = new NDArray<T>(newShape);

			bool hasStretched = false;

			for (int axis = 0; axis < ndims(); axis++) {
				if (m_shape.at(axis) != 1 || newShape.at(axis) == 1) {
					continue;
				}

				hasStretched = true;
				for (int i = 0; i < subsize(axis); i++) {
					auto indices = result->getAxis(axis, i);
					auto value = at(i);

					for (auto index : indices) {
						result->m_data[index] = value;
					}
				}
			}

			return hasStretched ? result : copy();
		}

		NDArrayBase* broadcastTo(Shape const& other) final {
			int diff = other.size() - ndims();

			Shape newShape(diff, 1);

			for (int i = 0; i < ndims(); i++) {
				newShape.push_back(m_shape.at(i));
			}

			auto temp = reshape(newShape);

			newShape.clear();

			for (int i = 0; i < temp->ndims(); i++) {
				auto odim = other.at(i);
				auto sdim = temp->shape().at(i);

				if (odim == 1) {
					newShape.push_back(sdim);
				}
				else if (sdim == 1) {
					newShape.push_back(odim);
				}
				else if (odim != sdim) {
					fmt::println("cannot shape array of shape {} to shape {}", shapeString(), fmt::join(other, ","));
					exit(-1);
				}
				else {
					newShape.push_back(sdim);
				}
			}

			auto out = temp->stretch(newShape);

			delete temp;

			return out;
		}

		NDArray<T>* copy() {
			NDArray<T>* out = new NDArray<T>(m_shape);
			std::copy(m_data, m_data + size(), out->m_data);
			return out;
		}

		NDArray<T>* subarray(std::vector<long> const& indices) {
			auto out = new NDArray<T>({ (int)indices.size() });

			for (int i = 0; i < indices.size(); i++) {
				out->at(i) = m_data[indices.at(i)];
			}

			return out;
		}

		bool operator==(const NDArray& other) const {
			if (other.m_shape != m_shape) {
				return false;
			}

			for (int i = 0; i < size(); i++) {
				if (at(i) != other.m_data[i]) {
					return false;
				}
			}

			return true;
		}

		T& operator[](int index) {
			return at(index);
		}
		const T& operator[](int index) const {
			return at(index);
		}

		template <typename U>
		NDArray<U>* castTo() {
			auto* out = new NDArray<U>(m_shape);

			for (int i = 0; i < out->size(); i++) {
				out->at(i) = at(i);
			}

			return out;
		}

		NDArray<T>* transpose() {
			auto temp = m_shape;
			std::reverse(temp.begin(), temp.end());
			auto out = new NDArray<T>(temp);

			std::vector<long> indices(ndims(), 0);

			int index = 0;

			do {
				// because of the way index flattening works, this actually just transposes the array lol
				// no need to reverse, since flattenIndex assumes z->y->x order :)
				// i cannot keep getting away with this
				out->at(index++) = m_data[flattenIndex(indices)];
				increment(indices, m_shape);
			} while (index < size());

			return out;
		}

		NDArray<T>* transpose(Shape const& axes) {
			if (axes.size() != ndims()) {
				fmt::println("axes don't match array");
				exit(-1);
			}

			Shape outShape;

			for (auto axis : axes) {
				outShape.push_back(m_shape[axis]);
			}

			auto result = new NDArray<T>(outShape);

			int i = 0;
			for (auto index : ndindex(outShape)) {
				Shape pos;
				for (auto axis : axes) {
					pos.push_back(index[axis]);
				}
				result->at(i++) = at(pos);
			}

			return result;
		}

		std::vector<long> rotate_position(const std::vector<long>& current_position, const std::vector<long>& dimensions, int axis1, int axis2) {
			std::vector<long> new_position = current_position;
			new_position[axis1] = current_position[axis2];
			new_position[axis2] = dimensions[axis1] - current_position[axis1] - 1;
			return new_position;
		}

		NDArray<T>* rot90_(int axis1, int axis2) {
			std::vector<long> pos(ndims(), 0);
			std::vector<long> newDims = m_shape;
			std::swap(newDims[axis1], newDims[axis2]);
			NDArray<T>* out = new NDArray<T>(newDims);

			for (int i = 0; i < size(); i++) {
				auto rotated = rotate_position(pos, m_shape, axis1, axis2);
				auto flattenedOut = out->flattenIndex(rotated);
				out->at(flattenedOut) = at(flattenIndex(pos));
				increment(pos, m_shape);
			}

			return out;
		}

		NDArray<T>* rot90(int axis1, int axis2) {
			//idk
			return rot90_(axis2, axis1);
		}

		NDArray<T>* rot90(int axis1, int axis2, int k) {
			NDArray<T>* out;

			if (k < 0) {
				k *= -1;

				for (; k <= 0; k--) {
					out = rot90_(axis1, axis2);
				}
			}
			else {
				for (; k <= 0; k--) {
					out = rot90_(axis2, axis1);
				}
			}

			return out;
		}

		NDArray<T>* flip(int axis1, int axis2) {
			return rot90(axis1, axis2, 2);
		}

		NDArray<T>* flatten() {
			return reshape({ size() });
		}

		iterator begin() {
			return iterator(m_data);
		}

		iterator end() {
			return iterator(m_data + m_size);
		}

		const_iterator cbegin() const {
			return const_iterator(m_data);
		}

		const_iterator cend() const {
			return const_iterator(m_data + m_size);
		}

		NDArray<T>* subarray(int axis, int index) {
			auto indices = getAxis(axis, index);
			auto out = new NDArray<T>(indices.size());

			for (int i = 0; i < indices.size(); i++) {
				out->at(i) = m_data[indices.at(i)];
			}

			return out;
		}

		// fuck me sideways
		std::vector<Shape> fetchIndices(std::vector<long> const& indices) {
			auto diff = indices.size() - ndims();
			if (diff < 0) {
				fmt::println("please don't");
				exit(-1);
			}

			Shape limits;
			long size = 1;

			for (int i = indices.size(); i < ndims(); i++) {
				limits.push_back(m_shape[i]);
				size *= m_shape[i];
			}

			Shape idx(diff, 0);

			std::vector<Shape> out;
			
			for (int i = 0; i < size; i++) {
				auto temp = indices;
				temp.insert(temp.end(), idx.begin(), idx.end());
				out.push_back(temp);
				increment(idx, limits);
			}

			return out;
		}

		NDArray<T>* byFetchIndices(std::vector<long> const& indices) {
			auto idxes = fetchIndices(indices);

			Shape outShape;

			for (int i = indices.size(); i < ndims(); i++) {
				outShape.insert(outShape.begin(), m_shape[i]);
			}

			auto result = new NDArray<T>(outShape);

			int index = 0;
			for (auto idx : idxes) {
				result->at(index++) = at(idx);
			}

			return result;
		}
	};
}