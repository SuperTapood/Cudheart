#pragma once

#include "../Inc.h"
#include "../Arrays/Arrays.h"

namespace Test {
	/// <summary>
	/// test the creation process of vectors
	/// </summary>
	void directArrayCreationTest();
	void creationFunctionsTest();
	/// <summary>
	/// Test the entire Cudheart module and print out progress and test results
	/// </summary>
	void test();
	/// <summary>
	/// assert expression and raise exception if false
	/// </summary>
	/// <param name="exp"> : bool - the bool value to check</param>
	void assert(bool exp, string a, string b);
}