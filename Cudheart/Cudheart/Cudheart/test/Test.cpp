#include "Test.h"
#include "../Exceptions/Exceptions.h"

void Test::directCreationTest()
{
	Vector vi = Vector(12);
	int empty = -842150451;
	for (int i = 0; i < vi.size; i++) {
		assert(*((int*)vi[i]) == empty, "vi[i]", "-842150451");
	}
	Vector bi = Vector(12, new DInt());
	cout << bi << endl;
	assert(vi == bi, "vi", "Vector(12, new DInt())");
	cout << "passed IntVector creation test" << endl;
}

void Test::test() {
	try {
		directCreationTest();
	}
	catch (BaseException& e) {
		e.print();
	}
}

void Test::assert(bool exp, string a, string b) {
	if (!exp) {
		throw AssertionError(a, b);
	}
}