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

		check("VectorOps::emptyLike<int>", a->getShape()->toString(), b->getShape()->toString());
	}

	void testArange() {
		auto a = arange(5);

		for (int i = 0; i < a->getSize(); i++) {
			check("VectorOps::arange<int>(int)", a->get(i), i);
		}

		auto b = arange(-1);
		check("VectorOps::arange<int>(int)", b->getShape()->toString(), "(0,)");

		auto c = arange(1, 5);
		check("VectorOps::arange<int>(int)", c->getShape()->toString(), "(0,)");

		auto d = arange(7, 5, 5);
		check("VectorOps::arange<int>(int)", d->getShape()->toString(), "(0,)");

		auto e = arange(5, 2);
		for (int i = 0, idx = 0; i < 5; i += 2, idx++) {
			check("VectorOps::arange<int>(int, int)", e->get(idx) == i);
		}

		auto f = arange(3, 7, 1);
		for (int i = 3, idx = 0; i < 7; i++, idx++) {
			check("VectorOps::arange<int>(int, int)", f->get(idx) == i);
		}
	}

	void testFull() {
		auto vec = full(5, 5);

		for (int i = 0; i < vec->getSize(); i++) {
			check("VectorOps::full<int>(int, int)", vec->get(i), 5);
		}
	}

	void testFullLike() {
		auto a = full(5, 5);
		auto b = fullLike(a, 5);

		check("VectorOps::fullLike(int, int)", a->getSize(), b->getSize());
		for (int i = 0; i < a->getSize(); i++) {
			check("VectorOps::fullLike(int, int)", a->get(i), b->get(i));
		}
	}

	void testLinspace() {
		auto a = linspace(5.f, 10.f);

		for (int i = 0; i < a->getSize(); i++) {
			check("VectorOps::linspace(float, float)", a->get(i), (float)(5 + (5.f / 49.f) * i));
		}
		
		
		auto b = linspace(7.f, 12.f, 2.f);

		for (int i = 0; i < b->getSize(); i++) {
			check("VectorOps::linspace(float, float, float)", b->get(i), (float)(7 + 5 * i));
		}

		auto c = linspace(5.f, 10.f, 10.f, false);

		for (int i = 0; i < c->getSize(); i++) {
			check("VectorOps::linspace(float, float, float, bool)", c->get(i), (float)(5 + 0.5 * i));
		}
	}

	void testOnes() {
		auto vec = ones<int>(5);

		for (int i = 0; i < vec->getSize(); i++) {
			check("VectorOps::ones<int>", vec->get(i), 1);
		}
	}

	void testOnesLike() {
		auto a = ones<int>(5);
		auto b = onesLike(a);

		check("VectorOps::onesLike<int>", a->getSize(), b->getSize());

		for (int i = 0; i < a->getSize(); i++) {
			check("VectorOps::onesLike<int>", a->get(i), b->get(i));
		}
	}

	void testZeros() {
		auto vec = zeros<int>(5);

		for (int i = 0; i < vec->getSize(); i++) {
			check("VectorOps::zeros<int>", vec->get(i), 0);
		}
	}

	void testZerosLike() {
		auto a = zeros<int>(5);
		auto b = zerosLike(a);

		check("VectorOps::zerosLike<int>", a->getSize(), b->getSize());

		for (int i = 0; i < a->getSize(); i++) {
			check("VectorOps::zerosLike<int>", a->get(i), b->get(i));
		}
	}

	void testLogspace() {
		auto a = logspace(5.f, 10.f);

		for (int i = 0; i < a->getSize(); i++) {
			check("VectorOps::logspace(float, float)", a->get(i), (float)pow(10, (float)(5.f + (5.f / 49.f) * i)));
		}


		auto b = logspace(7.f, 12.f, 2);

		for (int i = 0; i < b->getSize(); i++) {
			check("VectorOps::logspace(float, float, int)", b->get(i), (float)pow(10, (float)(7 + 5 * i)));
		}

		auto c = logspace(5.f, 10.f, 10, false);

		for (int i = 0; i < c->getSize(); i++) {
			check("VectorOps::logspace(float, float, int, bool)", c->get(i), (float)pow(10, (float)(5 + 0.5 * i)));
		}

		auto d = logspace(5.f, 10.f, 10, false, 2.f);

		for (int i = 0; i < d->getSize(); i++) {
			check("VectorOps::logspace(float, float, int, bool, float)", d->get(i), (float)pow(2, (float)(5 + 0.5 * i)));
		}
	}

	void testGeomspace() {
		
		auto a = geomspace(5.f, 10.f);

		for (int i = 0; i < a->getSize(); i++) {
			check("VectorOps::geomspace(float, float)", a->get(i), (float)pow(10.f, log10(5.f) + ((log10(10.f) - log10(5.f)) / 49.f) * (float)i));
		}


		auto b = geomspace(7.f, 12.f, 2);

		for (int i = 0; i < b->getSize(); i++) {
			check("VectorOps::geomspace(float, float, int)", b->get(i), (float)pow(10.f, (float)(log10(7.f) + ((log10(12.f) - log10(7.f))) * i)));
		}

		auto c = geomspace(5.f, 10.f, 10, false);

		for (int i = 0; i < c->getSize(); i++) {
			check("VectorOps::geomspace(float, float, int, bool)", c->get(i), (float)pow(10.f, (float)(log10(5.f) + ((log10(10.f) - log10(5.f)) / 10.f) * i)));
		}
	}
}