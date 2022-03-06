#include "ShapeError.h"

using Cudheart::Arrays::Shape;

ShapeError::ShapeError(string a, string b)
{
	ostringstream os;
	os << "ShapeError: cannot reshape array of shape " << a << " into shape " << b;
	msg = os.str();
}