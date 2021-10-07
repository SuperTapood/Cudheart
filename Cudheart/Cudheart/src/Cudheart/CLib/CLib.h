#pragma once


#include "../Core/core.h"
#include "CMath.h"
#include "CRandom.h"
#include "CVector.h"


class CLib : public Lib {
public:
	CLib() {
		math = CMath();
		rand = CRandom();
	}
};