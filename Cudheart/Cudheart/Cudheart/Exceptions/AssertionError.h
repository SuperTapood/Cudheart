#pragma once

#include "BaseException.h"

class AssertionError : public BaseException {
public:
	/// <summary>
	/// create an AssertionError object
	/// </summary>
	/// <param name="left"> : string - the left side of the expression</param>
	/// <param name="right"> : string - the right side of the expression</param>
	AssertionError(string left, string right);
};