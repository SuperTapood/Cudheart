#pragma once

#include "BaseException.cuh"

namespace Cudheart::Exceptions {
	class BadTypeException : public BaseException {
	public:
		BadTypeException(string expected, string got, bool autoraise = true) {
			ostringstream os;
			os << "BadTypeException: expected " << expected << " but got " << got << endl;
			m_msg = os.str();
			if (autoraise) {
				raise();
			}
		}

		BadTypeException(string msg, bool autoraise = true) {
			m_msg = msg;
			if (autoraise) {
				raise();
			}
		}
	};
}