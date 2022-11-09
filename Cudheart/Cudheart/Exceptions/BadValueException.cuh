#pragma once

#include "BaseException.cuh"

namespace Cudheart::Exceptions {
	class BadValueException : public BaseException {
	public:
		BadValueException(string funcName, string got, string exp, bool autoraise = true) {
			ostringstream os;
			os << "BadValueException: function " << funcName << " got " << got << " expected " << exp;
			m_msg = os.str();
			if (autoraise) {
				raise();
			}
		}
	};
}