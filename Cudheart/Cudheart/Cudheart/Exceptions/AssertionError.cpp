#include "AssertionError.h"


AssertionError::AssertionError(string left, string right) {
	ostringstream os;
	os << left << " is supposed to equal " << right << " but it doesn't.";
	msg = os.str();
}