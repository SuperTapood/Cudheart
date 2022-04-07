#pragma once

#include "BaseException.cuh"

namespace Cudheart::Exceptions {
	class NotImplementedException : public BaseException {
	public:
		NotImplementedException(string name, string dep) {
			ostringstream os;
			string exp = "is";
			for (int i = 0; i < dep.size(); i++) {
				if (dep.at(i) == ' ') {
					exp = "are";
					break;
				}
			}
			os << "function " << name << "is not implemented";
			if (dep != "") {
				os << " because " << dep << exp << " also not implemeted";
			}
			m_msg = os.str();
		}

		NotImplementedException(string name) : NotImplementedException(name, "") {}
	};
}