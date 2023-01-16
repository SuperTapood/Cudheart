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

	void testEye() {
		auto a = eye<int>(7, 6, 2);

		for (int i = 0; i < a->getHeight(); i++) {
			for (int j = 0; j < a->getWidth(); j++) {
				int comp = 0;

				if (i + 2 == j) {
					comp = 1;
				}

				check("MatrixOps::eye(int, int, int)", a->get(i, j), comp);
			}
		}

		auto b = eye<int>(7, 2);

		for (int i = 0; i < b->getHeight(); i++) {
			for (int j = 0; j < b->getWidth(); j++) {
				int comp = 0;

				if (i + 2 == j) {
					comp = 1;
				}

				check("MatrixOps::eye(int, int)", b->get(i, j), comp);
			}
		}

		auto c = eye<int>(4);

		for (int i = 0; i < c->getHeight(); i++) {
			for (int j = 0; j < c->getWidth(); j++) {
				check("MatrixOps::eye(int)", c->get(i, j), (int)(i == j));
			}
		}
	}

	void testMeshgrid() {
		auto base = arange(10, 5, 2);
		auto va = (Vector<int>*)base->flatten();
		auto vb = (Vector<int>*)base->flatten();

		Matrix<double>** mats = meshgrid<double>(va, vb);

		auto ma = mats[0];
		auto mb = mats[1];

		for (int i = 0; i < va->getSize(); i++) {
			for (int j = 0; j < vb->getSize(); j++) {
				check("MatrixOps::meshgrid(Vector<int>, Vector<int>)", ma->get(i, j), va->get(j));
			}
		}

		for (int i = 0; i < va->getSize(); i++) {
			for (int j = 0; j < vb->getSize(); j++) {
				check("MatrixOps::meshgrid", mb->get(i, j), vb->get(i));
			}
		}

		auto matss = meshgrid<int>(base, base);

		check("MatrixOps::meshgrid(Matrix<int>, Matrix<int>)", matss[0]->toString(), ma->toString());
		check("MatrixOps::meshgrid(Matrix<int>, Matrix<int>)", matss[1]->toString(), mb->toString());
	}

	void testDiag() {
		auto mat = arange(9, 3, 3);
		auto a = diag(mat, 1);

		for (int i = 0; i < a->getSize(); i++) {
			check("MatrixOps::diag(Matrix<int>, int)", a->get(i), mat->get(i, i + 1));
		}

		auto b = diag(mat);

		for (int i = 0; i < b->getSize(); i++) {
			check("MatrixOps::diag(Matrix<int>, int)", b->get(i), mat->get(i, i));
		}

		auto c = diagflat(mat->flatten(), 1);

		for (int i = 0; i < c->getHeight(); i++) {
			for (int j = 0; j < c->getWidth(); j++) {
				int comp = 0;

				if (i + 1 == j) {
					comp = mat->get(i);
				}

				check("MatrixOps::diagflat(Vector<int>, int)", c->get(i, j), comp);
			}
		}

		auto d = diagflat(mat->flatten());

		for (int i = 0; i < d->getHeight(); i++) {
			for (int j = 0; j < d->getWidth(); j++) {
				int comp = 0;

				if (i == j) {
					comp = mat->get(i);
				}

				check("MatrixOps::diagflat(Vector<int>)", d->get(i, j), comp);
			}
		}
	}

	void testTri() {
	}
}