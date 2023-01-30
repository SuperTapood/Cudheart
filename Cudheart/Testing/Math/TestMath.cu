#include "TestMath.cuh"

namespace Cudheart::Testing {
	void testMath() {
		Math::testBaseMath();
		Math::testBitwise();
	}
}