#include "Searching.cuh"

namespace Cudheart::Testing::Indexing::Searching {
	using namespace Cudheart::Searching;
	void test() {
		testArgmax();
		testArgmin();
		testNonzero();
		testArgwhere();
		testFlatnonzero();
		testWhere();
		testSearchsorted();
		testExtract();
		testCount_nonzero();
	}

	void testArgmax() {
		string cmd;

		int arr[] = {5, 2, 25, 25, 22, 21, 4, 7, 11};

		auto vec = new Vector(arr, 9);
		auto out = argmax(vec);

		cmd = Numpy::createArray("[5, 2, 25, 25, 22, 21, 4, 7, 11]", "vec");
		cmd += Numpy::argmax("vec", "res");

		Testing::submit("Searching::argmax(Vector<int>)", cmd, to_string(out));

	}

	void testArgmin() {
		string cmd;

		int arr[] = { 5, 2, 25, 25, 22, 21, 4, 7, 11 };

		auto vec = new Vector(arr, 9);
		auto out = argmin(vec);

		cmd = Numpy::createArray("[5, 2, 25, 25, 22, 21, 4, 7, 11]", "vec");
		cmd += Numpy::argmin("vec", "res");

		Testing::submit("Searching::argmin(Vector<int>)", cmd, to_string(out));
	}

	void testNonzero() {
		string cmd;

		int arr[] = { 5, 0, 25, 0, 22, 0, 4, 0, 11 };

		auto vec = new Vector(arr, 9);
		auto a = nonzero(vec);

		cmd = Numpy::createArray("[5, 0, 25, 0, 22, 0, 4, 0, 11]", "vec");
		cmd += Numpy::nonzero("vec", "res");

		Testing::submit("Searching::nonzero(Vector<int>)", cmd, a->toString());

		auto mat = (Matrix<int>*)vec->reshape<int>(new Shape(3, 3));
		auto b = nonzero(mat);

		cmd = Numpy::createArray("[5, 0, 25, 0, 22, 0, 4, 0, 11]", "vec");
		cmd += Numpy::reshape("vec", "(3, 3)", "mat");
		cmd += Numpy::nonzero("mat", "res");

		Testing::submit("Searching::nonzero(Matrix<int>)", cmd, b->toString());
	}

	void testArgwhere() {
		string cmd;
		
		auto vec = arange(6, 2, 3)->flatten();
		auto a = argwhere(vec);

		cmd = Numpy::arange("0", "6", "1", "int", "vec");
		cmd += Numpy::argwhere("vec", "res");

		Testing::submit("Searching::argwhere(Vector<int>)", cmd, a->toString());
	}

	void testFlatnonzero() {

	}

	void testWhere() {

	}

	void testSearchsorted() {

	}

	void testExtract() {

	}

	void testCount_nonzero() {

	}
}