﻿#include "Testing/Test.cuh"
#include "Misc/kernel.cuh"
#include "Misc/Test.cuh"
#include "Misc/unified.cuh"
#include "Misc/Custom.cuh"
#include "Misc/promotion.cuh"

#include <iostream>
#include <iomanip>

using namespace std;

// using Cudheart::Exceptions::CudaException;

int main() {
	// verify unified memory?
	// fmain();
	// compare speeds
	// func();
	// test inheritance with the ndarrays (do we even need that?)
	// testThing();
	// test everything
	testAll();
	// test templating with cuda
	// testCustom();
	// sillyTemplates();
	return 0;
}