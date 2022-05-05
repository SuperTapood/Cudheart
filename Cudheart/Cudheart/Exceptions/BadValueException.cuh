#pragma once

#include "BaseException.cuh"

namespace Cudheart::Exceptions {
	class BadValueException : public BaseException {
	public:
		BadValueException(string funcName, string got, string exp) {
			ostringstream os;
			os << "BadValueException: function " << funcName << " got " << got << " expected " << exp;
			m_msg = os.str();
			raise();
		}
	};
}