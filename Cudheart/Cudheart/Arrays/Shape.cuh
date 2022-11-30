#pragma once

namespace Cudheart::NDArrays {
	class Shape {
	private:
		int x, y;
		int size;
		int dims;

	public:
		Shape(int x, int y) {
			this->x = x;
			this->y = y;
			size = x * y;
			this->dims = 2;
		}

		Shape(int x) {
			this->x = x;
			this->y = 1;
			size = x;
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
			return size;
		}
	};
}