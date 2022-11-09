#pragma once

#include "BaseException.cuh"

namespace Cudheart::Exceptions {
	class ZeroDivisionException : public BaseException {
	public:
		ZeroDivisionException(string funcName, bool autoraise = true) {
			ostringstream os;
			os << "ZeroDivisionException: cannot compute " << funcName << "(0)";
			m_msg = os.str();
			if (autoraise) {
				raise();
			}
		}
	};
}