#pragma once

#include "BaseException.cuh"

namespace Cudheart::Exceptions {
	class MatrixConversionException : public BaseException {
	public:
		MatrixConversionException(int width, int height, int size) {
			ostringstream os;
			os << "expected matrix size of " << size << " got width " << width << " and height " << height << "\n";
			m_msg = os.str();
		}
	};
}