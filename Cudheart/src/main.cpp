#include <iostream>

#include "Cudheart/Cudheart.h"

void test() {
	creationTest();
}

void creationTest() {

}


int main() {
	std::printf("Cudheart using version %s\n", Cudheart::version.c_str());
	test();
	return 0;
}