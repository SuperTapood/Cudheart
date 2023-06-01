#pragma once
#include <string>

using namespace std;

namespace Cudheart {
	namespace NDArrays {
		class Shape {
		private:
			int x, y;
			int ndims;

		public:
			Shape(int x, int y) {
				this->x = x;
				this->y = y;
				this->ndims = 2;
			}

			Shape(int x) {
				this->x = x;
				this->y = 1;
				this->ndims = 1;
			}

			int getX() {
				return x;
			}

			int getY() {
				return y;
			}

			int getDims() {
				return ndims;
			}

			int size() {
				return x * y;
			}

			string toString() {
				ostringstream os;
				os << "(";
				os << x;
				os << ",";
				if (y != 1) {
					os << " ";
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