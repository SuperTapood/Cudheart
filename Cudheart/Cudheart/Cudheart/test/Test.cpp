#include "Test.h"
#include "../Exceptions/Exceptions.h"
#include "../Arrays/Shape.h"
#include "../Misc/GarbageCollector.h"

void Test::vectorCreationTest()
{
	Array ai = Array(new Shape(12));
	Array bi = Array(new Shape(12), new DInt());
	int arr[] = {5, 140, 1410, 250, 5};
	Array ci = Array((void*)arr, new Shape(56), new DInt());
	for (int i = 0; i < 5; i++) {
		int v = DInt().cast(ci.get(i));
		assert(v == arr[i], "" + v, "" + arr[i]);
	}
	cout << "passed int array creation test" << endl;
}

void Test::test() {
	try {
		vectorCreationTest();
	}
	catch (BaseException& e) {
		e.print();
		GarbageCollector::destroy();
	}
}

void Test::assert(bool exp, string a, string b) {
	if (!exp) {
		throw AssertionError(a, b);
	}
}