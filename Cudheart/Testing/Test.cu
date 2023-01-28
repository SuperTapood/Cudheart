#include "Test.cuh"
#include "TestUtil.cuh"
#include <chrono>
#include <ctime>

void testAll() {
	auto start = std::chrono::system_clock::now();
	Cudheart::Testing::testArrays();
	Cudheart::Testing::testIndexing();
	Cudheart::Testing::testLogic();
	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed = end - start;
	Cudheart::Testing::Testing::get()->testAll();
	// cout << "all tests passed in " << elapsed.count() << "s\n";
}