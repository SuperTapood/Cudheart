#include "IndexError.h"

IndexError::IndexError(size_t got, size_t boundA, size_t boundB) {
	ostringstream os;
	os << "IndexError: Got index " << got << ", expected index between " << boundA << " and " << boundB;
	msg = os.str();
}

//IndexError::IndexError(Shape* a, Shape* b)
//{
//	ostringstream os;
//	os << "IndexError: cannot use shape " << (*a).toString() << " to index array with shape " << (*b).toString();
//	msg = os.str();
//}
//
//IndexError::IndexError(size_t len, Shape* s)
//{
//	ostringstream os;
//	os << "IndexError: ndim index of length " << len << " cannot be used to index array of shape " << (*s).toString() << " and length " << (*s).length;
//	msg = os.str();
//}