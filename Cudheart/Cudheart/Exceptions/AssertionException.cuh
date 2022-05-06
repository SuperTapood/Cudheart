#pragma once

#include "BaseException.cuh"

namespace Cudheart::Exceptions {
	template <typename T, typename U>
	class AssertionException : public BaseException {
	public:
		AssertionException(string funcName, T a, U b) {
			ostringstream os;
			os << "AssertionException: function " << funcName << " asserted " << a << " == " << b;
			m_msg = os.str();
		}
	};
}