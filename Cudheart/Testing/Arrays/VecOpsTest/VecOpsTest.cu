#include "VecOpsTest.cuh"

namespace Cudheart::Testing::Arrays::VecOpsTest {
	using namespace Cudheart::VectorOps;

	void test() {
		testEmptyLike();
		testArange();
		testFull();
		testFullLike();
		testLinspace();
		testOnes();
		testOnesLike();
		testZeros();
		testZerosLike();
		testLogspace();
		testGeomspace();
	}

	void testEmptyLike() {
		auto a = new Vector<int>(5);
		auto b = emptyLike<int>(a);
		string cmd;

		cmd = Numpy::empty("(5,)", "res");
		cmd += "res = res.shape";

		Testing::submit("VectorOps::emptyLike<int>", cmd, b->getShape()->toString());
	}

	void testArange() {
		string cmd;

		auto a = arange(5);

		cmd = Numpy::arange("0", "5", "1", "int", "res");

		Testing::submit("VectorOps::arange<int>(int)", cmd, a->toString());

		auto b = arange(5, 2);

		cmd = Numpy::arange("0", "5", "2", "int", "res");

		Testing::submit("VectorOps::arange<int>(int, int)", cmd, b->toString());

		auto c = arange(3, 7, 1);

		cmd = Numpy::arange("3", "7", "1", "int", "res");

		Testing::submit("VectorOps::arange<int>(int, int, int)", cmd, c->toString());
	}

	void testFull() {
		auto vec = full(5, 5);

		string cmd = Numpy::full("(5,)", "5", "res");

		Testing::submit("VectorOps::full<int>(int, int)", cmd, vec->toString());
	}

	void testFullLike() {
		auto a = full(5, 5);
		auto b = fullLike(a, 5);

		string cmd = Numpy::full("(5,)", "5", "res");

		Testing::submit("VectorOps::fullLike(int, int)", cmd, b->toString());
	}

	void testLinspace() {
		string cmd;

		auto a = linspace<long double>(5.0, 10.0);

		cmd = Numpy::linspace("5", "10", "50", "True", "float", "res");

		Testing::submit("VectorOps::linspace(float, float)", cmd, a->toString());

		auto b = linspace(7.0, 12.0, 2.0);

		cmd = Numpy::linspace("7", "12", "2", "True", "float", "res");

		Testing::submit("VectorOps::linspace(float, float, float)", cmd, b->toString());

		auto c = linspace(5.0, 10.0, 10.0, false);

		cmd = Numpy::linspace("5", "10", "10", "False", "float", "res");

		Testing::submit("VectorOps::linspace(float, float, float, bool)", cmd, c->toString());
	}

	void testOnes() {
		auto vec = ones<int>(5);

		string cmd = Numpy::full("(5,)", "1", "res");

		Testing::submit("VectorOps::ones<int>", cmd, vec->toString());
	}

	void testOnesLike() {
		auto a = ones<int>(5);
		auto b = onesLike(a);

		string cmd = Numpy::full("(5,)", "1", "res");

		Testing::submit("VectorOps::onesLike<int>", cmd, b->toString());
	}

	void testZeros() {
		auto vec = zeros<int>(5);

		string cmd = Numpy::full("(5,)", "0", "res");

		Testing::submit("VectorOps::zeros<int>", cmd, vec->toString());
	}

	void testZerosLike() {
		auto a = zeros<int>(5);
		auto b = zerosLike(a);

		string cmd = Numpy::full("(5,)", "0", "res");

		Testing::submit("VectorOps::zerosLike<int>", cmd, b->toString());
	}

	void testLogspace() {
		string cmd;

		auto a = logspace(5.f, 10.f);

		cmd = Numpy::logspace("5", "10", "50", "True", "10", "float", "res");

		Testing::submit("VectorOps::logspace(float, float)", cmd, a->toString());

		auto b = logspace(7.f, 12.f, 2);

		cmd = Numpy::logspace("7", "12", "2", "True", "10", "float", "res");

		Testing::submit("VectorOps::logspace(float, float, int)", cmd, b->toString());

		auto c = logspace(5.f, 10.f, 10, false);

		cmd = Numpy::logspace("5", "10", "10", "False", "10", "float", "res");

		Testing::submit("VectorOps::logspace(float, float, int, bool)", cmd, c->toString());

		auto d = logspace(5.f, 10.f, 10, false, 2.f);

		cmd = Numpy::logspace("5", "10", "10", "False", "2", "float", "res");

		Testing::submit("VectorOps::logspace(float, float, int, bool, float)", cmd, d->toString());
	}

	void testGeomspace() {
		string cmd;

		auto a = geomspace(5.f, 10.f);

		cmd = Numpy::geomspace("5", "10", "50", "True", "float", "res");

		Testing::submit("VectorOps::geomspace(float, float)", cmd, a->toString());

		auto b = geomspace(7.f, 12.f, 2);

		cmd = Numpy::geomspace("7", "12", "2", "True", "float", "res");

		Testing::submit("VectorOps::geomspace(float, float, int)", cmd, b->toString());

		auto c = geomspace(5.f, 10.f, 10, false);

		cmd = Numpy::geomspace("5", "10", "10", "False", "float", "res");

		Testing::submit("VectorOps::geomspace(float, float, int, bool)", cmd, c->toString());
	}
}