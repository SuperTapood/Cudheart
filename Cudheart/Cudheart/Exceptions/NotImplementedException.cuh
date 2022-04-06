#pragma once

#include "BaseException.cuh"

namespace Cudheart::Exceptions {
	class NotImplementedException : public BaseException {
	public:
		NotImplementedException(string name, string dep) {
			ostringstream os;
			os << "function " << name << "is not implemented";
			if (dep != "") {
				os << " because " << dep << "is also not implemeted";
			}
			m_msg = os.str();
		}

		NotImplementedException(string name) : NotImplementedException(name, "") {}
	};
}