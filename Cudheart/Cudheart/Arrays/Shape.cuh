#pragma once
#include <string>

using namespace std;

namespace Cudheart {
	namespace NDArrays {
		class Shape {
		private:
			int x, y;
			int dims;

		public:
			Shape(int x, int y) {
				this->x = x;
				this->y = y;
				this->dims = 2;
			}

			Shape(int x) {
				this->x = x;
				this->y = 1;
				this->dims = 1;
			}

			int getX() {
				return x;
			}

			int getY() {
				return y;
			}

			int getDims() {
				return dims;
			}

			int getSize() {
				return x * y;
			}

			string toString() {
				ostringstream os;
				os << "(";
				os << x;
				os << ", ";
				if (y != 1) {
					os << y;
				}
				os << ")";
				return os.str();
			}

			void print() {
				cout << toString() << endl;
			}
		};
	}
}