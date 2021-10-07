#include <stdio.h>
#include <iostream>

#include "test.h"

#include "../Utils/utils.h"
#include "../CLib/CLib.h"

using namespace Cudheart::CObjects;


void Cudheart::Test::test() {
	//exceptionTest();
	VectorTest vt = VectorTest();
	vt.creationTest();
}

void Cudheart::Test::exceptionTest() {
	Cudheart::Utils::Exceptions::BadShape::raise(5, 2);
}


void Cudheart::Test::VectorTest::creationTest() {
	int arr[] = { 5, 5, 5 };
	int shape[] = { 3 };
	CVector<int> v = CVector<int>(arr, Shape(shape), Dtype());
}