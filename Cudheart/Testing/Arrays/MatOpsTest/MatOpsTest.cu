#include "MatOpsTest.cuh"

namespace Cudheart::Testing::Arrays::MatOpsTest {
	using namespace Cudheart::MatrixOps;

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
		auto a = new Matrix<int>(5, 5);
		auto b = emptyLike<int>(a);

		check("MatrixOps::emptyLike<int>", a->getShape()->toString(), b->getShape()->toString());
	}

	void testArange() {
		auto a = arange(6, 2, 3);

		for (int i = 0; i < a->getSize(); i++) {
			check("MatrixOps::arange<int>(int)", a->get(i), i);
		}

		auto e = arange(0, 5, 2, 3, 1);
		for (int i = 0, idx = 0; i < 5; i += 2, idx++) {
			check("MatrixOps::arange<int>(int, int)", e->get(idx) == i);
		}

		auto f = arange(3, 7, 1, 2, 2);
		for (int i = 3, idx = 0; i < 7; i++, idx++) {
			check("MatrixOps::arange<int>(int, int)", f->get(idx) == i);
		}
	}

	void testFull() {
		auto mat = full(5, 5, 5);

		for (int i = 0; i < mat->getSize(); i++) {
			check("MatrixOps::full<int>(int, int)", mat->get(i), 5);
		}
	}

	void testFullLike() {
		auto a = full(5, 5, 5);
		auto b = fullLike(a, 5);

		check("MatrixOps::fullLike(int, int)", a->getShape()->toString(), b->getShape()->toString());
		for (int i = 0; i < a->getSize(); i++) {
			check("MatrixOps::fullLike(int, int)", a->get(i), b->get(i));
		}
	}

	void testLinspace() {
		auto a = linspace(5.f, 10.f, 5, 10);

		for (int i = 0; i < a->getSize(); i++) {
			check("MatrixOps::linspace(float, float)", a->get(i), (float)(5 + (5.f / 49.f) * i));
		}


		auto b = linspace(7.f, 12.f, 2.f, 2, 1);

		for (int i = 0; i < b->getSize(); i++) {
			check("MatrixOps::linspace(float, float, float)", b->get(i), (float)(7 + 5 * i));
		}

		auto c = linspace(5.f, 10.f, 10.f, false, 5, 2);

		for (int i = 0; i < c->getSize(); i++) {
			check("MatrixOps::linspace(float, float, float, bool)", c->get(i), (float)(5 + 0.5 * i));
		}
	}

	void testOnes() {
		auto mat = ones<int>(5, 5);

		for (int i = 0; i < mat->getSize(); i++) {
			check("MatrixOps::ones<int>", mat->get(i), 1);
		}
	}

	void testOnesLike() {
		auto a = ones<int>(5, 5);
		auto b = onesLike(a);

		check("MatrixOps::onesLike<int>", a->getSize(), b->getSize());

		for (int i = 0; i < a->getSize(); i++) {
			check("MatrixOps::onesLike<int>", a->get(i), b->get(i));
		}
	}

	void testZeros() {
		auto mat = zeros<int>(5, 5);

		for (int i = 0; i < mat->getSize(); i++) {
			check("MatrixOps::zeros<int>", mat->get(i), 0);
		}
	}

	void testZerosLike() {
		auto a = zeros<int>(5, 5);
		auto b = zerosLike(a);

		check("MatrixOps::zerosLike<int>", a->getSize(), b->getSize());

		for (int i = 0; i < a->getSize(); i++) {
			check("MatrixOps::zerosLike<int>", a->get(i), b->get(i));
		}
	}

	void testLogspace() {
		auto a = logspace(5.f, 10.f, 5, 10);

		for (int i = 0; i < a->getSize(); i++) {
			check("MatrixOps::logspace(float, float)", a->get(i), (float)pow(10, (float)(5.f + (5.f / 49.f) * i)));
		}


		auto b = logspace(7.f, 12.f, 2.f, 1, 2);

		for (int i = 0; i < b->getSize(); i++) {
			check("MatrixOps::logspace(float, float, int)", b->get(i), (float)pow(10, (float)(7 + 5 * i)));
		}

		auto c = logspace(5.f, 10.f, 10.f, false, 5, 2);

		for (int i = 0; i < c->getSize(); i++) {
			check("MatrixOps::logspace(float, float, int, bool)", c->get(i), (float)pow(10, (float)(5 + 0.5 * i)));
		}

		auto d = logspace(5.f, 10.f, 10.f, false, 2.f, 5, 2);

		for (int i = 0; i < d->getSize(); i++) {
			check("MatrixOps::logspace(float, float, int, bool, float)", d->get(i), (float)pow(2, (float)(5 + 0.5 * i)));
		}
	}

	void testGeomspace() {

		auto a = geomspace(5.f, 10.f, 25, 2);

		for (int i = 0; i < a->getSize(); i++) {
			check("MatrixOps::geomspace(float, float)", a->get(i), (float)pow(10.f, log10(5.f) + ((log10(10.f) - log10(5.f)) / 49.f) * (float)i));
		}


		auto b = geomspace(7.f, 12.f, 2.f, 2, 1);

		for (int i = 0; i < b->getSize(); i++) {
			check("MatrixOps::geomspace(float, float, int)", b->get(i), (float)pow(10.f, (float)(log10(7.f) + ((log10(12.f) - log10(7.f))) * i)));
		}

		auto c = geomspace(5.f, 10.f, 10.f, false, 2, 5);

		for (int i = 0; i < c->getSize(); i++) {
			check("MatrixOps::geomspace(float, float, int, bool)", c->get(i), (float)pow(10.f, (float)(log10(5.f) + ((log10(10.f) - log10(5.f)) / 10.f) * i)));
		}
	}
}