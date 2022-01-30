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

	Array crr = ArrayOps::arange(16);
	for (int i = 0; i < crr.size; i++) {
		assert(i == (*(double*)crr[i]), to_string(i), to_string(*(double*)crr[i]));
	}

	Array drr = ArrayOps::arange(50);
	drr.reshape(new Shape(5, 10));
	DDouble d = DDouble();
	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < 10; j++) {
			assert(i * 10 + j == d.cast(drr.get(2, i, j)), to_string(10 * i + j), to_string(d.cast(drr.get(2, i, j))));
		}
	}

	Array err = ArrayOps::empty(new Shape(5, 14, 520));
	Array frr = ArrayOps::emptyLike(&err);
	assert(err == frr, "err", "frr");

	Array grr = ArrayOps::ones(new Shape(2547, 44));
	Array hrr = ArrayOps::onesLike(&grr);
	assert(grr == hrr, "grr", "hrr");

	Array irr = ArrayOps::zeros(new Shape(5250));
	Array jrr = ArrayOps::zerosLike(&irr);
	assert(irr == jrr, "irr", "jrr");

	int value = 69420;
	Array krr = ArrayOps::full(new Shape(5, 14, 520), &value);
	Array lrr = ArrayOps::fullLike(&krr, &value);
	assert(krr == lrr, "krr", "lrr");

	// fix this not working
	// but like later

	int fr[]{ 1, 2, 3, 4 };
	int* fg = (int*)fr;
	void* fa = (void*)fg;
	int er[]{ 1, 2, 3, 4, 5};
	int* eg = (int*)er;
	void* ea = (void*)eg;
	Array f = ArrayOps::asarray(fa, new Shape(4), new DInt());
	Array e = ArrayOps::asarray(ea, new Shape(5), new DInt());
	Array* meshes = ArrayOps::meshgrid(&f, &e);
	cout << meshes[0].toString() << endl;
	cout << meshes[1].toString() << endl;

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