#pragma once

#include "BaseException.cuh"

namespace Cudheart::Exceptions {
	class BadTypeException : public BaseException {
	public:
		BadTypeException(string expected, string got) {
			ostringstream os;
			os << "BadTypeException: expected " << expected << " but got " << got << endl;
			m_msg = os.str();
		}
	};
}