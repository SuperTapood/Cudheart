#pragma once

#include "BaseException.cuh"

namespace Cudheart::Exceptions {
	class ShapeMismatchException : public BaseException {
	public:
		ShapeMismatchException(int a, int b) {
			ostringstream os;
			os << "ShapeMismatchException: vector of size " << a << " does not match vector of size " << b;
			m_msg = os.str();
		}

		ShapeMismatchException(int aw, int ah, int bw, int bh) {
			ostringstream os;
			os << "ShapeMismatchException: matrix of width " << aw << " and height " << ah << " does not match matrix of width " << bw << " and height " << bh;
			m_msg = os.str();
		}

		ShapeMismatchException(string custom) {
			m_msg = "ShapeMismatchException: " + custom;
		}
	};
}