#include "TestIndexing.cuh"

namespace Cudheart::Testing {
	void testIndexing() {
		// auto start = std::chrono::system_clock::now();
		// tests are sorted in a hierarchical order
		// dependencies will be tested first, then the modules that depend on them
		// this is done to simplify debugging upon feature break
		Indexing::Searching::test();
		Indexing::Sorting::test();
		// auto end = std::chrono::system_clock::now();
		// std::chrono::duration<double> elapsed = end - start;
		// cout << "array tests passed in " << elapsed.count() << "s\n";
	}
}