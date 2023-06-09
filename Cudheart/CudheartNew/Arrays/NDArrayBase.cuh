#pragma once

namespace CudheartNew {
	class NDArrayBase {
	protected:
		std::vector<long> m_shape;
		long m_size = 1;

	public:
		long size() const {
			return m_size;
		}

		long subsize(int axis) {
			long out = 1;

			for (int i = axis + 1; i < ndims(); i++) {
				out *= m_shape.at(i);
			}

			return out;
		}

		std::vector<long> subshape(int axis) {
			std::vector<long> out;

			for (int i = 0; i < ndims(); i++) {
				if (i == axis) {
					continue;
				}
				out.push_back(m_shape.at(i));
			}

			return out;
		}

		int ndims() const {
			return m_shape.size();
		}

		std::string shapeString() const {
			if (m_shape.size() == 0) {
				return "()";
			}
			return fmt::format("({})", fmt::join(m_shape, ","));
		}

		virtual std::string printRecursive(long* s, int len, int start, int offset) = 0;

		std::string toString(bool verbose = false) {
			std::vector<long> arr = m_shape;
			std::string out = printRecursive(arr.data(), ndims(), 0, 0);
			return verbose ? fmt::format("{}, shape={}", out, shapeString()) : out;
		}

		void println(bool verbose = false) {
			fmt::println(toString(verbose));
		}

		std::vector<long> shape() {
			return m_shape;
		}

		NDArrayBase* broadcastTo(NDArrayBase* other) {
			return broadcastTo(other->shape());
		}

		virtual NDArrayBase* broadcastTo(std::vector<long> const& other) = 0;
	};
}