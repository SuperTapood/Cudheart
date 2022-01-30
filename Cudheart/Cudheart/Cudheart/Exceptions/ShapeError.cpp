#include "pch.h"
#include "ShapeError.h"

ShapeError::ShapeError(Shape* a, Shape* b)
{
	ostringstream os;
	os << "ShapeError: cannot reshape array of shape " << (*a).toString() << " into shape " << (*b).toString();
	msg = os.str();
}
