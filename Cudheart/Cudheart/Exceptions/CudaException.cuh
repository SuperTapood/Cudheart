#pragma once

#include "BaseException.cuh"


namespace Cudheart::Exceptions {
	class CudaException : public BaseException {
	public:
		CudaException(const char* name, const char* full) {
			ostringstream os;
			os << "CudaException: " << name << " " << full << endl;
			m_msg = os.str();
		}
	};
}