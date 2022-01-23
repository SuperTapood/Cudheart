#pragma once


#include "BaseException.h"

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
};