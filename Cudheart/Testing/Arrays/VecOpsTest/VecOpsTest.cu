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

		check("VectorOps::emptyLike<int>", b->toString());
	}

	void testArange() {
		auto a = arange(5);

		check("VectorOps::arange<int>(int)", a->toString());

		auto b = arange(5, 2);

		check("VectorOps::arange<int>(int, int)", b->toString());

		auto c = arange(3, 7, 1);

		check("VectorOps::arange<int>(int, int, int)", c->toString());
	}

	void testFull() {
		auto vec = full(5, 5);

		check("VectorOps::full<int>(int, int)", vec->toString());
	}

	void testFullLike() {
		auto a = full(5, 5);
		auto b = fullLike(a, 5);

		check("VectorOps::fullLike(int, int)", b->toString());
	}

	void testLinspace() {
		auto a = linspace(5.f, 10.f);

		check("VectorOps::linspace(float, float)", a->toString());

		auto b = linspace(7.f, 12.f, 2.f);

		check("VectorOps::linspace(float, float, float)", b->toString());

		auto c = linspace(5.f, 10.f, 10.f, false);

		check("VectorOps::linspace(float, float, float, bool)", c->toString());
	}

	void testOnes() {
		auto vec = ones<int>(5);

		check("VectorOps::ones<int>", vec->toString());
	}

	void testOnesLike() {
		auto a = ones<int>(5);
		auto b = onesLike(a);

		check("VectorOps::onesLike<int>", b->toString());
	}

	void testZeros() {
		auto vec = zeros<int>(5);

		check("VectorOps::zeros<int>", vec->toString());
	}

	void testZerosLike() {
		auto a = zeros<int>(5);
		auto b = zerosLike(a);

		check("VectorOps::zerosLike<int>", b->toString());
	}

	void testLogspace() {
		auto a = logspace(5.f, 10.f);

		check("VectorOps::logspace(float, float)", a->toString());

		auto b = logspace(7.f, 12.f, 2);

		check("VectorOps::logspace(float, float, int)", b->toString());

		auto c = logspace(5.f, 10.f, 10, false);

		check("VectorOps::logspace(float, float, int, bool)", c->toString());

		auto d = logspace(5.f, 10.f, 10, false, 2.f);

		check("VectorOps::logspace(float, float, int, bool, float)", d->toString());
	}

	void testGeomspace() {
		auto a = geomspace(5.f, 10.f);

		check("VectorOps::geomspace(float, float)", a->toString());

		auto b = geomspace(7.f, 12.f, 2);

		check("VectorOps::geomspace(float, float, int)", b->toString());

		auto c = geomspace(5.f, 10.f, 10, false);

		check("VectorOps::geomspace(float, float, int, bool)", c->toString());
	}
}