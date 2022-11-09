#include "Cudheart/Testing/Test.cuh"
#include "Cudheart/Cudheart.cuh"
#include "Misc/kernel.cuh"
#include "Misc/Test.cuh"
#include "Misc/unified.cuh"
#include "Cudheart/Exceptions/FileNotFoundException.cuh"

using Cudheart::Exceptions::CudaException;

int main() {
	// verify unified memory?
	// fmain();
	// compare speeds
	// func();
	// test inheritance with the ndarrays (do we even need that?)
	// testThing();
	// test everything
	testAll();
	return 0;
}