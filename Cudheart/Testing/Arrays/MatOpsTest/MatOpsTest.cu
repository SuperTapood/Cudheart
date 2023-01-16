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
		testEye();
		testMeshgrid();
		testDiag();
		testTri();
	}

	void testEmptyLike() {
		auto a = new Matrix<int>(5, 5);
		auto b = emptyLike<int>(a);

		check("MatrixOps::emptyLike<int>", b->toString());
	}

	void testArange() {
		auto a = arange(6, 2, 3);

		check("MatrixOps::arange<int>(int)", a->toString());

		auto e = arange(0, 5, 2, 3, 1);

		check("MatrixOps::arange<int>(int, int)", e->toString());

		auto f = arange(3, 7, 1, 2, 2);

		check("MatrixOps::arange<int>(int, int)", f->toString());
	}

	void testFull() {
		auto mat = full(5, 5, 5);

		check("MatrixOps::full<int>(int, int)", mat->toString());
	}

	void testFullLike() {
		auto a = full(5, 5, 5);
		auto b = fullLike(a, 5);

		check("MatrixOps::fullLike(int, int)", b->toString());
	}

	void testLinspace() {
		auto a = linspace(5.f, 10.f, 5, 10);

		check("MatrixOps::linspace(float, float)", a->toString());

		auto b = linspace(7.f, 12.f, 2.f, 2, 1);
		
		check("MatrixOps::linspace(float, float, float)", b->toString());

		auto c = linspace(5.f, 10.f, 10.f, false, 5, 2);

		check("MatrixOps::linspace(float, float, float, bool)", c->toString());
	}

	void testOnes() {
		auto mat = ones<int>(5, 5);

		check("MatrixOps::ones<int>", mat->toString());
	}

	void testOnesLike() {
		auto a = ones<int>(5, 5);
		auto b = onesLike(a);

		check("MatrixOps::onesLike<int>", b->toString());
	}

	void testZeros() {
		auto mat = zeros<int>(5, 5);

		check("MatrixOps::zeros<int>", mat->toString());
	}

	void testZerosLike() {
		auto a = zeros<int>(5, 5);
		auto b = zerosLike(a);

		check("MatrixOps::zerosLike<int>", b->toString());
	}

	void testLogspace() {
		auto a = logspace(5.f, 10.f, 5, 10);

		check("MatrixOps::logspace(float, float, int, int)", a->toString());

		auto b = logspace(7.f, 12.f, 2.f, 1, 2);

		check("MatrixOps::logspace(float, float, float, int, int)", b->toString());

		auto c = logspace(5.f, 10.f, 10.f, false, 5, 2);

		check("MatrixOps::logspace(float, float, float, bool, int, int)", c->toString());

		auto d = logspace(5.f, 10.f, 10.f, false, 2.f, 5, 2);

		check("MatrixOps::logspace(float, float, float, bool, float, int, int)", d->toString());
	}

	void testGeomspace() {
		auto a = geomspace(5.f, 10.f, 25, 2);

		check("MatrixOps::geomspace(float, float, int, int)", a->toString());

		auto b = geomspace(7.f, 12.f, 2.f, 2, 1);

		check("MatrixOps::geomspace(float, float, float, int, int)", b->toString());

		auto c = geomspace(5.f, 10.f, 10.f, false, 2, 5);

		check("MatrixOps::geomspace(float, float, float, bool, int, int)", c->toString());
	}

	void testEye() {
		auto a = eye<int>(7, 6, 2);

		check("MatrixOps::eye(int, int, int)", a->toString());

		auto b = eye<int>(7, 2);

		check("MatrixOps::eye(int, int)", b->toString());

		auto c = eye<int>(4);

		check("MatrixOps::eye(int)", c->toString());
	}

	void testMeshgrid() {
		auto base = arange(10, 5, 2);
		auto va = (Vector<int>*)base->flatten();
		auto vb = (Vector<int>*)base->flatten();

		auto mats = meshgrid<double>(va, vb);

		auto ma = mats->get(0);
		auto mb = mats->get(1);

		check("MatrixOps::meshgrid(Vector<int>, Vector<int>)", ma->toString());
		check("MatrixOps::meshgrid(Vector<int>, Vector<int>)", mb->toString());

		auto matss = meshgrid<int>(base, base);

		check("MatrixOps::meshgrid(Matrix<int>, Matrix<int>)", matss->get(0)->toString());
		check("MatrixOps::meshgrid(Matrix<int>, Matrix<int>)", matss->get(1)->toString());
	}

	void testDiag() {
		auto mat = arange(9, 3, 3);
		auto a = diag(mat, 1);

		check("MatrixOps::diag(Matrix<int>, int)", a->toString());

		auto b = diag(mat);

		check("MatrixOps::diag(Matrix<int>)", b->toString());

		auto c = diagflat(mat->flatten(), 1);

		check("MatrixOps::diag(Vector<int>, int)", c->toString());

		auto d = diagflat(mat->flatten());

		check("MatrixOps::diagflat(Matrix<int>)", d->toString());
	}

	void testTri() {
		auto a = tri<int>(2, 2, 1);
		a->print();
	}
}