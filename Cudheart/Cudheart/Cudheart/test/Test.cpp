#include "Test.h"
#include "../Exceptions/Exceptions.h"
#include "../Arrays/Shape.h"

void Test::directCreationTest()
{
	Array vi = Array(new Shape({ 12 }));
	cout << vi << endl;
	Array bi = Array(new Shape({ 12 }), new DInt());
	cout << bi << endl;
	cout << "passed int array creation test" << endl;
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