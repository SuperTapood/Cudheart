#pragma once


#include "../Core/core.h"
#include "../Clib/CMath.h"
#include "../CLib/CRandom.h"


class CLib : public Lib {
public:
	CLib() {
		math = CMath();
		rand = CRandom();
	}
};