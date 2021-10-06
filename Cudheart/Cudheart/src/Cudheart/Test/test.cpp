#include <stdio.h>
#include <iostream>

#include "test.h"

#include "../Utils/utils.h"

using namespace std;


void Cudheart::Test::test() {
	//exceptionTest();
	VectorTest vt = VectorTest();
	vt.creationTest();
}

void Cudheart::Test::exceptionTest() {
	Cudheart::Utils::Exceptions::BadShape::raise(5, 2);
}


void Cudheart::Test::VectorTest::creationTest() {

}