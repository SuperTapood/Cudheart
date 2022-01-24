#include "Test.h"
#include "../Exceptions/Exceptions.h"
#include "../Arrays/Shape.h"

void Test::directArrayCreationTest()
{
	Array ai = Array(new Shape(12));
	Array bi = Array(new Shape(12), new DInt());
	int arr[] = {5, 140, 1410, 250, 5};
	Array ci = Array((void*)arr, new Shape(5), new DInt());
	for (int i = 0; i < 5; i++) {
		int v = DInt().cast(ci.get(i));
		assert(v == arr[i], to_string(v), "" + to_string(arr[i]));
	}
	int brr[] = { 785, 578, 27, 22350, 5728 };
	Array di = Array((void*)brr, new Shape(5));
	for (int i = 0; i < 5; i++) {
		int v = DInt().cast(di.get(i));
		assert(v == brr[i], to_string(v), to_string(brr[i]));
	}
	cout << "passed int array creation test" << endl;
}

void Test::creationFunctionsTest() {
	// asarray and empty here
	cout << "passed creation functions test" << endl;
}

void Test::test() {
	try {
		// expand this and add more tests
		directArrayCreationTest();
		creationFunctionsTest();
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