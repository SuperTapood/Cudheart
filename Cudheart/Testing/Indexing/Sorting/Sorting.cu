#include "Sorting.cuh"

namespace Cudheart::Testing::Indexing::Sorting {
	using namespace Cudheart::Sorting;
	using namespace Cudheart::NDArrays;

	void test() {
		testQuicksort();
		testMergesort();
		testHeapsort();
		testSort();
		testArgsort();
		/*testPartition();
		testArgpartition();*/
	}

	void testQuicksort() {
		string cmd;

		Vector<int>* vec = new Vector({ 1, 25, 5820, 5, 58, 22, 48, 14, 32, 22, 99, 90900, 15, 11 });
		NDArray<int>* a = quicksort(vec);
		
		cmd = Numpy::createArray("[1, 25, 5820, 5, 58, 22, 48, 14, 32, 22, 99, 90900, 15, 11]", "vec");
		cmd += Numpy::sort("vec", "'quicksort'", "res");

		Testing::submit("Sorting::quicksort(Vector<int>)", cmd, a->toString());
	}

	void testMergesort() {
		string cmd;

		Vector<int>* vec = new Vector({ 1, 25, 5820, 5, 58, 22, 48, 14, 32, 22, 99, 90900, 15, 11 });
		NDArray<int>* a = mergesort(vec);

		cmd = Numpy::createArray("[1, 25, 5820, 5, 58, 22, 48, 14, 32, 22, 99, 90900, 15, 11]", "vec");
		cmd += Numpy::sort("vec", "'mergesort'", "res");

		Testing::submit("Sorting::mergesort(Vector<int>)", cmd, a->toString());
	}

	void testHeapsort() {
		string cmd;

		Vector<int>* vec = new Vector({ 1, 25, 5820, 5, 58, 22, 48, 14, 32, 22, 99, 90900, 15, 11 });
		NDArray<int>* a = heapsort(vec);

		cmd = Numpy::createArray("[1, 25, 5820, 5, 58, 22, 48, 14, 32, 22, 99, 90900, 15, 11]", "vec");
		cmd += Numpy::sort("vec", "'heapsort'", "res");

		Testing::submit("Sorting::heapsort(Vector<int>)", cmd, a->toString());
	}

	void testSort() {
		string cmd;

		Vector<int>* vec = new Vector({ 1, 25, 5820, 5, 58, 22, 48, 14, 32, 22, 99, 90900, 15, 11 });
		NDArray<int>* a = sort(vec, Kind::Quicksort);

		cmd = Numpy::createArray("[1, 25, 5820, 5, 58, 22, 48, 14, 32, 22, 99, 90900, 15, 11]", "vec");
		cmd += Numpy::sort("vec", "'quicksort'", "res");

		Testing::submit("Sorting::sort(Vector<int>, Kind::Quicksort)", cmd, a->toString());

		NDArray<int>* b = sort(vec, Kind::Mergesort);

		cmd = Numpy::createArray("[1, 25, 5820, 5, 58, 22, 48, 14, 32, 22, 99, 90900, 15, 11]", "vec");
		cmd += Numpy::sort("vec", "'mergesort'", "res");

		Testing::submit("Sorting::sort(Vector<int>, Kind::Mergesort)", cmd, b->toString());

		NDArray<int>* c = sort(vec, Kind::Heapsort);

		cmd = Numpy::createArray("[1, 25, 5820, 5, 58, 22, 48, 14, 32, 22, 99, 90900, 15, 11]", "vec");
		cmd += Numpy::sort("vec", "'heapsort'", "res");

		Testing::submit("Sorting::sort(Vector<int>, Kind::Heapsort)", cmd, c->toString());

		NDArray<int>* d = sort(vec);

		cmd = Numpy::createArray("[1, 25, 5820, 5, 58, 22, 48, 14, 32, 22, 99, 90900, 15, 11]", "vec");
		cmd += Numpy::sort("vec", "'quicksort'", "res");

		Testing::submit("Sorting::sort(Vector<int>)", cmd, d->toString());
	}

	void testArgsort() {
		string cmd;

		Vector<int>* vec = new Vector({ 1, 25, 5820, 5, 58, 22, 48, 14, 32, 22, 99, 90900, 15, 11 });

		NDArray<int>* a = argsort(vec, Kind::Quicksort);

		cmd = Numpy::createArray("[1, 25, 5820, 5, 58, 22, 48, 14, 32, 22, 99, 90900, 15, 11]", "vec");
		cmd += Numpy::argsort("vec", "'quicksort'", "res");

		Testing::submit("Sorting::argsort(Vector<int>, Kind::Quicksort)", cmd, a->toString());

		NDArray<int>* b = argsort(vec, Kind::Mergesort);

		cmd = Numpy::createArray("[1, 25, 5820, 5, 58, 22, 48, 14, 32, 22, 99, 90900, 15, 11]", "vec");
		cmd += Numpy::argsort("vec", "'mergesort'", "res");

		Testing::submit("Sorting::argsort(Vector<int>, Kind::Mergesort)", cmd, b->toString());

		NDArray<int>* c = argsort(vec, Kind::Heapsort);

		cmd = Numpy::createArray("[1, 25, 5820, 5, 58, 22, 48, 14, 32, 22, 99, 90900, 15, 11]", "vec");
		cmd += Numpy::argsort("vec", "'heapsort'", "res");

		Testing::submit("Sorting::argsort(Vector<int>, Kind::Heapsort)", cmd, c->toString());

		NDArray<int>* d = argsort(vec);

		cmd = Numpy::createArray("[1, 25, 5820, 5, 58, 22, 48, 14, 32, 22, 99, 90900, 15, 11]", "vec");
		cmd += Numpy::argsort("vec", "'quicksort'", "res");

		Testing::submit("Sorting::argsort(Vector<int>)", cmd, d->toString());
	}
}