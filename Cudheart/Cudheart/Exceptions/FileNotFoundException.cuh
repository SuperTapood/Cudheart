#pragma once

#include "BaseException.cuh"

namespace Cudheart {
	namespace Exceptions {
		class FileNotFoundException : public BaseException {
		public:
			FileNotFoundException(string fileName, bool autoraise = true) {
				ostringstream os;
				os << "FileNotFoundException: file named " << fileName << " could not be located";
				m_msg = os.str();
				if (autoraise) {
					raise();
				}
			}
		};
	}
}