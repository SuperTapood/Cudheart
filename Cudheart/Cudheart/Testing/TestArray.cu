#include "TestArray.cuh"

void testArray() {
	testIO();
}


void testIO() {
	using namespace Cudheart::IO;
	string str = "1 2 3 4";

	auto a = fromString(str);
	a->print();
}