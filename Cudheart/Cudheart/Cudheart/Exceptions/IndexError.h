#pragma once


#include "BaseException.h"

class IndexError : public BaseException {
public:
	IndexError(int got, int boundA, int boundB) {
		ostringstream os;
		os << "IndexError: Got index " << got << ", expected index between " << boundA << " and " << boundB;
		msg = os.str();
	}
};