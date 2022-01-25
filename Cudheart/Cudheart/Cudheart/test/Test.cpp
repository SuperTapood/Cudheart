#include "Test.h"
#include "../Exceptions/Exceptions.h"
#include "../Arrays/Shape.h"

void Test::directIntVectorCreation() {
	Array ai = Array(new Shape(12));
	Array bi = Array(new Shape(12), new DInt());
	int arr[] = { 5, 140, 1410, 250, 5 };
	Array ci = Array((void*)arr, new Shape(5), new DInt());
	for (int i = 0; i < 5; i++) {
		int v = DInt().cast(ci.get(1, i));
		assert(v == arr[i], to_string(v), "" + to_string(arr[i]));
	}
	int brr[] = { 785, 578, 27, 22350, 5728 };
	Array di = Array((void*)brr, new Shape(5));
	for (int i = 0; i < 5; i++) {
		int v = DInt().cast(di.get(1, i));
		assert(v == brr[i], to_string(v), to_string(brr[i]));
	}
	cout << "passed int vector creation test" << endl;
}

void Test::directDoubleVectorCreation() {
	Array bi = Array(new Shape(12), new DDouble());
	double arr[] = { 5.454, 140.54475, 14107837.5, 250.25, 5.444 };
	Array ci = Array((void*)arr, new Shape(5), new DDouble());
	for (int i = 0; i < 5; i++) {
		double v = DDouble().cast(ci.get(1, i));
		assert(v == arr[i], to_string(v), "" + to_string(arr[i]));
	}
	cout << "passed double vector creation test" << endl;
}

void Test::directArrayCreationTest()
{
	directIntVectorCreation();
	directDoubleVectorCreation();
}

void Test::creationFunctionsTest() {
	int a[]{5, 5, 5401, 14, 25402};
	Array arr = ArrayOps::asarray(a, new Shape(5));
	Array brr = ArrayOps::asarray(a, new Shape(5), new DInt());
	DInt dtype = DInt();
	for (int i = 0; i < 5; i++) {
		assert(dtype.cast(arr[i]) == a[i], to_string(dtype.cast(arr[i])), to_string(a[i]));
		assert(dtype.cast(brr[i]) == a[i], to_string(dtype.cast(brr[i])), to_string(a[i]));
		assert(dtype.cast(arr[i]) == dtype.cast(brr[i]), to_string(dtype.cast(arr[i])), to_string(dtype.cast(brr[i])));
	}

	cout << "passed creation functions test" << endl;
}

void Test::test() {
	try {
		// expand this and add more tests
		// directArrayCreationTest();
		// creationFunctionsTest();
		//Array ai = ArrayOps::arange(300);
		//ai.reshape(new Shape(3, 5, 20));
		//cout << DDouble().cast(ai.get(3, 1, 1, 1)) << endl;
		//ai.reshape(new Shape(5, 20, 3));
		//cout << DDouble().cast(ai.get(3, 1, 1, 1)) << endl;
		cout << ArrayOps::eye(2).toString() << endl;
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