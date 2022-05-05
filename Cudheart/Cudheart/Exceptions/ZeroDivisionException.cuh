#pragma once

#include "BaseException.cuh"

namespace Cudheart::Exceptions {
	class ZeroDivisionException : public BaseException {
	public:
		ZeroDivisionException(string funcName) {
			ostringstream os;
			os << "ZeroDivisionException in " << funcName << "()";
			m_msg = os.str();
			raise();
		}
	};
}