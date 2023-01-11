#include "Test.cuh"
#include <chrono>
#include <ctime>

void testAll() {
	auto start = std::chrono::system_clock::now();
	Cudheart::Testing::testArrays();
	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed = end - start;
	// cout << "all tests passed in " << elapsed.count() << "s\n";
}