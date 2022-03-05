#include "Test.h"
#include "../Cudheart.h"


namespace Cudheart::Test {
	using namespace Cudheart::Arrays;

	void TestCreation() {

	}

	void Test() {
		TestCreation();
	}

	void Assert(bool exp, string a, string b) {
		if (!exp) {
			throw AssertionError(a, b);
		}
	}
}