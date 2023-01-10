#pragma once

#include "BaseException.cuh"

namespace Cudheart {
	namespace Exceptions {
		class ShapeMismatchException : public BaseException {
		public:
			ShapeMismatchException(int a, int b, bool autoraise = true) {
				ostringstream os;
				os << "ShapeMismatchException: vector of size " << a << " does not match vector of size " << b;
				m_msg = os.str();
				if (autoraise) {
					raise();
				}
			}

			ShapeMismatchException(int aw, int ah, int bw, int bh, bool autoraise = true) {
				ostringstream os;
				os << "ShapeMismatchException: matrix of width " << aw << " and height " << ah << " does not match matrix of width " << bw << " and height " << bh;
				m_msg = os.str();
				if (autoraise) {
					raise();
				}
			}

			ShapeMismatchException(string custom, bool autoraise = true) {
				m_msg = "ShapeMismatchException: " + custom;
				if (autoraise) {
					raise();
				}
			}

			ShapeMismatchException(string s1, string s2, bool autoraise = true) {
				ostringstream os;
				os << "ShapeMismatchException: cannot reshape array of shape " << s1 << " into shape " << s2;
				m_msg = os.str();
				if (autoraise) {
					raise();
				}
			}
		};
	}
}