#pragma once

namespace Cudheart {
	class ComplexType {
	public:
		double real;
		double imag;

	public:
		ComplexType() {
			this->real = 0;
			this->imag = 0;
		}

		ComplexType(double real, double imag) {
			this->real = real;
			this->imag = imag;
		}
	};
}