#pragma once

#include <stdio.h>

namespace Cudheart {
	class ComplexType {
	public:
		double real;
		double imag;

	public:
		ComplexType(double real, double imag) {
			this->real = real;
			this->imag = imag;
		}

		template <typename T>
		ComplexType(T real) {
			if constexpr (is_arithmetic_v<T>) {
				this->real = real;
				this->imag = 0;
			}
			else {
				this->real = 0;
				this->imag = 0;
			}
		}

		ComplexType() : ComplexType(0, 0) {}

		string toString() {
			string r = to_string(real);
			string i = to_string(5);
			return r + " " + i + "i";
		}
	};
}