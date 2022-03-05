#pragma once

#include "BaseException.h"
#include "../Arrays/Shape.h"

using namespace Cudheart::Arrays;

class ShapeError : public BaseException {
public:
	ShapeError(Shape* a, Shape* b);
};