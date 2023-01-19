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
		testDiagFlat();
		testTri();
	}

	void testEmptyLike() {
		auto a = new Matrix<int>(5, 5);
		auto b = emptyLike<int>(a);
		
		string cmd = "res = [5, 5]\n";

		Testing::check("MatrixOps::emptyLike<int>", cmd, b->getShape()->toString());
	}

	void testArange() {
		string cmd;

		auto a = arange(6, 2, 3);

		cmd = Numpy::arange("0", "6", "1", "int", "res");
		cmd += Numpy::reshape("res", "(3, 2)", "res");

		Testing::check("MatrixOps::arange<int>(int)", cmd, a->toString());

		auto e = arange(0, 5, 2, 3, 1);

		cmd = Numpy::arange("0", "5", "2", "int", "res");
		cmd += Numpy::reshape("res", "(1, 3)", "res");

		Testing::check("MatrixOps::arange<int>(int, int)", cmd, e->toString());

		auto f = arange(3, 7, 1, 2, 2);

		cmd = Numpy::arange("3", "7", "1", "int", "res");
		cmd += Numpy::reshape("res", "(2, 2)", "res");

		Testing::check("MatrixOps::arange<int>(int, int)", cmd, f->toString());
	}

	void testFull() {
		auto mat = full(5, 5, 5);

		string cmd = Numpy::full("(5, 5)", "5", "res");

		Testing::check("MatrixOps::full<int>(int, int)", cmd, mat->toString());
	}

	void testFullLike() {
		auto a = full(5, 5, 5);
		auto b = fullLike(a, 5);

		string cmd = Numpy::full("(5, 5)", "5", "res");

		Testing::check("MatrixOps::fullLike(int, int)", cmd, b->toString());
	}

	void testLinspace() {
		string cmd;

		auto a = linspace(5.f, 10.f, 5, 10);

		cmd = Numpy::linspace("5", "10", "50", "True", "float", "res");
		cmd += Numpy::reshape("res", "(5, 10)", "res");

		Testing::check("MatrixOps::linspace(float, float)", cmd, a->toString());

		auto b = linspace(7.f, 12.f, 2.f, 2, 1);

		cmd = Numpy::linspace("7", "12", "2", "True", "float", "res");
		cmd += Numpy::reshape("res", "(2, 1)", "res");
		
		Testing::check("MatrixOps::linspace(float, float, float)", cmd, b->toString());

		auto c = linspace(5.f, 10.f, 10.f, false, 5, 2);

		cmd = Numpy::linspace("5", "10", "10", "False", "float", "res");
		cmd += Numpy::reshape("res", "(5, 2)", "res");

		Testing::check("MatrixOps::linspace(float, float, float, bool)", cmd, c->toString());
	}

	void testOnes() {
		auto mat = ones<int>(5, 5);

		string cmd = Numpy::full("(5, 5)", "1", "res");

		Testing::check("MatrixOps::ones<int>", cmd, mat->toString());
	}

	void testOnesLike() {
		auto a = ones<int>(5, 5);
		auto b = onesLike(a);

		string cmd = Numpy::full("(5, 5)", "1", "res");

		Testing::check("MatrixOps::onesLike<int>", cmd, b->toString());
	}

	void testZeros() {
		auto mat = zeros<int>(5, 5);

		string cmd = Numpy::full("(5, 5)", "0", "res");

		Testing::check("MatrixOps::zeros<int>", cmd, mat->toString());
	}

	void testZerosLike() {
		auto a = zeros<int>(5, 5);
		auto b = zerosLike(a);

		string cmd = Numpy::full("(5, 5)", "0", "res");

		Testing::check("MatrixOps::zerosLike<int>", cmd, b->toString());
	}

	void testLogspace() {
		string cmd;
		auto a = logspace(5.f, 10.f, 5, 10);
		
		cmd = Numpy::logspace("5", "10", "50", "True", "10", "float", "res");
		cmd += Numpy::reshape("res", "(5, 10)", "res");

		Testing::check("MatrixOps::logspace(float, float, int, int)", cmd, a->toString());

		auto b = logspace(7.f, 12.f, 2.f, 1, 2);

		cmd = Numpy::logspace("7", "12", "2", "True", "10", "float", "res");
		cmd += Numpy::reshape("res", "(1, 2)", "res");

		Testing::check("MatrixOps::logspace(float, float, float, int, int)", cmd, b->toString());

		auto c = logspace(5.f, 10.f, 10.f, false, 5, 2);

		cmd = Numpy::logspace("5", "10", "10", "False", "10", "float", "res");
		cmd += Numpy::reshape("res", "(5, 2)", "res");

		Testing::check("MatrixOps::logspace(float, float, float, bool, int, int)", cmd, c->toString());

		auto d = logspace(5.f, 10.f, 10.f, false, 2.f, 5, 2);

		cmd = Numpy::logspace("5", "10", "10", "False", "2", "float", "res");
		cmd += Numpy::reshape("res", "(5, 2)", "res");

		Testing::check("MatrixOps::logspace(float, float, float, bool, float, int, int)", cmd, d->toString());
	}

	void testGeomspace() {
		string cmd;
		auto a = geomspace(5.f, 10.f, 25, 2);

		cmd = Numpy::geomspace("5", "10", "50", "True", "float", "res");
		cmd += Numpy::reshape("res", "(25, 2)", "res");

		Testing::check("MatrixOps::geomspace(float, float, int, int)", cmd, a->toString());

		auto b = geomspace(7.f, 12.f, 2.f, 2, 1);

		cmd = Numpy::geomspace("7", "12", "2", "True", "float", "res");
		cmd += Numpy::reshape("res", "(2, 1)", "res");

		Testing::check("MatrixOps::geomspace(float, float, float, int, int)", cmd, b->toString());

		auto c = geomspace(5.f, 10.f, 10.f, false, 2, 5);

		cmd = Numpy::geomspace("5", "10", "10", "False", "float", "res");
		cmd += Numpy::reshape("res", "(2, 5)", "res");

		Testing::check("MatrixOps::geomspace(float, float, float, bool, int, int)", cmd, c->toString());
	}

	void testEye() {
		string cmd;
		auto a = eye<int>(7, 6, 2);

		cmd = Numpy::eye("7", "6", "2", "int", "res");

		Testing::check("MatrixOps::eye(int, int, int)", cmd, a->toString());

		auto b = eye<int>(7, 2);

		cmd = Numpy::eye("7", "7", "2", "int", "res");

		Testing::check("MatrixOps::eye(int, int)", cmd, b->toString());

		auto c = eye<int>(4);

		cmd = Numpy::eye("4", "4", "0", "int", "res");

		Testing::check("MatrixOps::eye(int)", cmd, c->toString());
	}

	void testMeshgrid() {
		string cmd;
		auto base = arange(10, 5, 2);
		auto va = (Vector<int>*)base->flatten();
		auto vb = (Vector<int>*)base->flatten();

		auto mats = meshgrid<double>(va, vb);

		cmd =  Numpy::arange("0", "10", "1", "int", "a");
		cmd += Numpy::arange("0", "10", "1", "int", "b");
		cmd += Numpy::meshgrid("a", "b", "res");
		cmd += "comp = np.array([" + mats[0]->toString() + ", " + mats[1]->toString() + "])";

		Testing::check("MatrixOps::meshgrid(Vector<int>, Vector<int>)", cmd, "comp");

		auto matss = meshgrid<int>(base, base);

		cmd =  Numpy::arange("0", "10", "1", "int", "a");
		cmd += Numpy::arange("0", "10", "1", "int", "b");
		cmd += Numpy::reshape("a", "(5, 2)", "a");
		cmd += Numpy::reshape("b", "(5, 2)", "b");
		cmd += Numpy::meshgrid("a", "b", "res");
		cmd += "comp = np.array([" + matss[0]->toString() + ", " + matss[1]->toString() + "])";

		Testing::check("MatrixOps::meshgrid(Matrix<int>, Matrix<int>)", cmd, "comp");
	}

	void testDiag() {
		auto mat = arange(9, 3, 3);
		string cmd = Numpy::arange("0", "9", "1", "int", "mat");
		cmd += Numpy::reshape("mat", "(3, 3)", "mat");
		string add;

		auto a = diag(mat, 1);

		add = Numpy::diag("mat", "1", "res");

		Testing::check("MatrixOps::diag(Matrix<int>, int)", cmd + add, a->toString());

		auto b = diag(mat);

		add = Numpy::diag("mat", "0", "res");

		Testing::check("MatrixOps::diag(Matrix<int>)", cmd + add, b->toString());
	}

	void testDiagFlat() {
		auto mat = arange(9, 3, 3);

		string cmd = Numpy::arange("0", "9", "1", "int", "mat");
		cmd += Numpy::reshape("mat", "(3, 3)", "mat");
		string add;

		auto a = diagflat(mat->flatten(), 1);

		add = Numpy::diagflat("mat", "1", "res");

		Testing::check("MatrixOps::diagflat(Vector<int>, int)", cmd + add, a->toString());

		auto b = diagflat(mat->flatten());

		add = Numpy::diagflat("mat", "0", "res");

		Testing::check("MatrixOps::diagflat(Vector<int>)", cmd + add, b->toString());
	}

	void testTri() {
		auto a = tri<int>(2, 2, 1);
		a->print();
	}
}