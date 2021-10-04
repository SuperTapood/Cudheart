#pragma once


#include "../Core/core.h"
#include "cmath.h"


class CLib : public Lib {
public:
	CLib() {
		math = CMath();
		// crand + calloc
	}
};