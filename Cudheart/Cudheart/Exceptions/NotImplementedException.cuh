#pragma once

#include "BaseException.cuh"

namespace Cudheart {
	namespace Exceptions {
		class NotImplementedException : public BaseException {
		public:
			NotImplementedException(string name, string dep, bool autoraise = true) {
				ostringstream os;
				string exp = "is ";
				for (int i = 0; i < dep.size(); i++) {
					if (dep.at(i) == ' ') {
						exp = "are";
						break;
					}
				}
				os << "function " << name << " is not implemented";
				if (dep != "") {
					os << " because " << dep << exp << " also not implemeted";
				}
				m_msg = os.str();
				if (autoraise) {
					raise();
				}
			}

			NotImplementedException(string name, bool autoraise = true) : NotImplementedException(name, "", autoraise) {}
		};
	}
}