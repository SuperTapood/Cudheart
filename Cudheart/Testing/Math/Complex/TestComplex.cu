#include "TestComplex.cuh"

namespace Cudheart::Testing::Math {
	void testComplex() {
		ComplexType::testComplexType();
		CPP::ComplexMath::testComplexMath();
	}
}