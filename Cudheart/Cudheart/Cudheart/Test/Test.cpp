#include "Test.h"
#include "../Cudheart.h"


namespace Cudheart::Test {
	using namespace Cudheart::Arrays;

	void TestCreation() {
		Array<int> a = *Arrays::ArrayOps<int>::zeros(new Shape(new int[]{1, 1, 2, 2, 3}, 2));
		print(a);
	}

	void Test() {
		TestCreation();
	}

	void Assert(bool exp, string a, string b) {
		if (!exp) {
			throw AssertionError(a, b);
		}
	}

	void print(auto msg) {
		cout << msg << endl;
	}
}