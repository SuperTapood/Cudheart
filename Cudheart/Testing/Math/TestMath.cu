#include "TestMath.cuh"

namespace Cudheart::Testing {
	void testMath() {
		Math::testBaseMath();
		Math::testBitwise();
		Math::testComplex();
		Math::testExps();
		Math::testLinalg();
		Math::testStatistics();
	}
}