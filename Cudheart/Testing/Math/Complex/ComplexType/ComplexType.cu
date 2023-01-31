#include "ComplexType.cuh"

namespace Cudheart::Testing::Math::ComplexType {

	using Cudheart::ComplexType;

	void testComplexType() {
		testConstructors();
	}

	void testConstructors() {
		string cmd;

		auto a = new ComplexType(5, 5);

		cmd = "res = 5+5j";

		Testing::submit("ComplexType(int, int)", cmd, a->toString());

		auto b = new ComplexType(5);

		cmd = "res = 5";

		Testing::submit("ComplexType(int)", cmd, b->toString());
	}
}