#include <iostream>

#include "Cudheart/Cudheart.h"

using namespace Cudheart;

void creationTest() {
	int arr[2] = { 3, 4 };
	CudaArray<int> c(arr);
}

void test() {
	creationTest();
}


int main() {
	std::printf("Cudheart using version %s\n", Cudheart::version.c_str());
	test();
	return 0;
}