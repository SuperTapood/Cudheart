#include "Test.h"
#include "../Exceptions/Exceptions.h"
#include "../Arrays/Shape.h"

void Test::directIntVectorCreation() {
	Array<int> ai = Array<int>(new Shape(12));
	Array<int> bi = Array<int>(new Shape(12));
	int arr[] = { 5, 140, 1410, 250, 5 };
	Array<int> ci = Array<int>(arr, new Shape(5));
	for (int i = 0; i < 5; i++) {
		int v = ci.get(1, i);
		assert(v == arr[i], to_string(v), "" + to_string(arr[i]));
	}
	int brr[] = { 785, 578, 27, 22350, 5728 };
	Array<int> di = Array<int>(brr, new Shape(5));
	for (int i = 0; i < 5; i++) {
		int v = di.get(1, i);
		assert(v == brr[i], to_string(v), to_string(brr[i]));
	}
	cout << "passed int vector creation test" << endl;
}

void Test::directDoubleVectorCreation() {
	Array<double> bi = Array<double>(new Shape(12));
	double arr[] = { 5.454, 140.54475, 14107837.5, 250.25, 5.444 };
	Array<double> ci = Array<double>(arr, new Shape(5));
	for (int i = 0; i < 5; i++) {
		double v = ci[i];
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
	Array<int> arr = *ArrayOps<int>::asarray(a, new Shape(5));
	Array<int> brr = *ArrayOps<int>::asarray(a, new Shape(5));
	for (int i = 0; i < 5; i++) {
		assert(arr[i] == a[i], to_string(arr[i]), to_string(a[i]));
		assert(brr[i] == a[i], to_string(brr[i]), to_string(a[i]));
		assert(arr[i] == brr[i], to_string(arr[i]), to_string(brr[i]));
	}

	/*Array<double> crr = ArrayOps<double>::arange(16);
	for (int i = 0; i < crr.size; i++) {
		assert(i == (crr[i]), to_string(i), to_string(crr[i]));
	}*/

	Array<double> drr = *ArrayOps<double>::arange(50);
	cout << drr.getShapeString() << endl;
	drr.reshape(new Shape(5, 10));
	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < 10; j++) {
			assert(i * 10 + j == drr.get(2, i, j), to_string(10 * i + j), to_string(drr.get(2, i, j)));
		}
	}

	Array<int> err = *ArrayOps<int>::empty(new Shape(5, 14, 520));
	Array<int> frr = *ArrayOps<int>::emptyLike(&err);
	// lmao you can't compare two emptys its not how this works you don't know what values they have
	// assert(err == frr, "err", "frr");

	Array<int> grr = *ArrayOps<int>::ones(new Shape(4, 4));
	Array<int> hrr = *ArrayOps<int>::onesLike(&grr);
	assert(grr == hrr, "grr", "hrr");

	Array<int> irr = *ArrayOps<int>::zeros(new Shape(5250));
	Array<int> jrr = *ArrayOps<int>::zerosLike(&irr);
	assert(irr == jrr, "irr", "jrr");

	int value = 69420;
	Array<int> krr = *ArrayOps<int>::full(new Shape(5, 14, 520), value);
	Array<int> lrr = *ArrayOps<int>::fullLike(&krr, value);
	assert(krr == lrr, "krr", "lrr");

	int fr[]{ 1, 2, 3, 4 };
	int er[]{ 1, 2, 3, 4, 5};
	Array<int> f = *ArrayOps<int>::asarray(fr, new Shape(4));
	Array<int> e = *ArrayOps<int>::asarray(er, new Shape(5));
	Array<int>* meshes = ArrayOps<int>::meshgrid(&f, &e);
	// cout << meshes[0].toString() << endl;
	// cout << meshes[1].toString() << endl;
	Array<int> aril = *ArrayOps<int>::tril(&meshes[0]);
	// cout << aril.toString() << endl;
	Array<int> bril = *ArrayOps<int>::tril(&meshes[1], 2);
	// cout << bril.toString() << endl;
	Array<int> cril = *ArrayOps<int>::tril(&meshes[1], -2);
	// cout << cril.toString() << endl;

	Array<int> ariu = *ArrayOps<int>::triu(&meshes[0]);
	// cout << ariu.toString() << endl;
	Array<int> briu = *ArrayOps<int>::triu(&meshes[1], 2);
	// cout << briu.toString() << endl;
	Array<int> criu = *ArrayOps<int>::triu(&meshes[1], -2);
	// cout << criu.toString() << endl;
	// add tests for eye, linspace, meshgrid and tril
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