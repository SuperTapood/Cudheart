#include "IndexError.h"

IndexError::IndexError(int got, int boundA, int boundB) {
	ostringstream os;
	os << "IndexError: Got index " << got << ", expected index between " << boundA << " and " << boundB;
	msg = os.str();
}