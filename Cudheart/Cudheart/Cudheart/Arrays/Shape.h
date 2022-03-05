#pragma once

#include "../Util.h"
#include "../Exceptions/Exceptions.h"

namespace Cudheart::Arrays {
	class Shape {
	public:
		int* dims;
		int len;
		int size;
		
	public:
		Shape(int dims[], int len);
		string toString();
	};
}