#include "NotImplementedError.h"


NotImplementedError::NotImplementedError(string name) {
	ostringstream os;
	os << "NotImplementedError: Function " << name << " is not implemented\nmaybe in a later build...";
	msg = os.str();
}