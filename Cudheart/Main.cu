
#include "Cudheart/Test/Test.cuh"
#include "Cudheart/Exceptions/Exceptions.cuh"

using Cudheart::Exceptions::CudaException;

int main() {
	try {
		test();
	}
	catch (const CudaException& e) {
		cout << e.what() << endl;
	}
}