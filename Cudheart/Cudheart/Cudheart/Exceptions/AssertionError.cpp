#include "AssertionError.h"


AssertionError::AssertionError(string left, string right) {
	ostringstream os;
	os << "Assertion Error: " << left << " is supposed to equal " << right << " but it doesn't.";
	msg = os.str();
}

AssertionError::AssertionError()
{
	msg = "Assertion failed";
}
