#pragma once

#include <string>
#include <stdio.h>

using namespace std;

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
			ostringstream os;
			os << real;
			os << "+";
			os << imag;
			os << "j";
			return os.str();
		}
	};
}