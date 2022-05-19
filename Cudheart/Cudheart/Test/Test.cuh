#pragma once

#include "../../Cudheart/Util.cuh"
#include "../../Cudheart/Cudheart.cuh"

template <typename T, typename U>
void assertTest(string func, T a, U b) {
	if (a != b) {
		AssertionException(func, a, b);
	}
}

void testExceptions();
void testVectorCreation();
void testMatrixCreation();
void testBinaryOpsCPP();
void testLinalgOpsCPP();
void testLogicOpsCPP();
void test();