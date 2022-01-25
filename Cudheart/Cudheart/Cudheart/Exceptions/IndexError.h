#pragma once


#include "BaseException.h"
#include "../Arrays/Shape.h"

/// <summary>
/// An exception indicating that an index provided is exceeding array bounds
/// </summary>
class IndexError : public BaseException {
public:
	/// <summary>
	/// create a new IndexError object
	/// </summary>
	/// <param name="got"> : int - the index exceeding the bounds</param>
	/// <param name="boundA"> : int - the lower bound</param>
	/// <param name="boundB"> : int - the higher bound</param>
	IndexError(int got, int boundA, int boundB);
	IndexError(Shape* a, Shape* b);
	IndexError(int len, Shape* s);
};