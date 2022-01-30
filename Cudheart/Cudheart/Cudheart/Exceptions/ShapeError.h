#pragma once

#include "BaseException.h"
#include "../Arrays/Shape.h"

class ShapeError : public BaseException {
public:
	ShapeError(Shape* a, Shape* b);
};