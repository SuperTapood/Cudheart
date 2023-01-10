#pragma once

#include "BaseException.cuh"

namespace Cudheart {
	namespace Exceptions {
		class MatrixConversionException : public BaseException {
		public:
			MatrixConversionException(int width, int height, int size, bool autoraise = true) {
				ostringstream os;
				os << "expected matrix size of " << size << " got width " << width << " and height " << height << "\n";
				m_msg = os.str();
				if (autoraise) {
					raise();
				}
			}
		};
	}
}