#include "LinalgCPP.cuh"

namespace Cudheart::Testing::Math::CPP::Linalg {
	using namespace Cudheart::CPP::Math::Linalg;

	void testLinalg() {
		testDot();
		testInner();
		testOuter();
		testDet();
		testTrace();
		testSolve();
		testRoots();
		testInv();
		testConvolve();
		testClip();
	}

	void testDot() {
		string cmd;

		auto a = arange(6, 2, 3);
		auto b = arange(3, 3, 1)->flatten();
		auto res = dot(a, b);

		cmd = Numpy::createArray(a->toString(), "a");
		cmd += Numpy::createArray(b->toString(), "b");
		cmd += Numpy::dot("a", "b", "res");

		Testing::submit("Linalg::dot(Matrix<int>, Vector<int>)", cmd, res->toString());

		auto c = (Matrix<int>*)a->reshape<int>(new Shape(3, 2));
		res = dot(b, c);

		cmd = Numpy::createArray(c->toString(), "a");
		cmd += Numpy::createArray(b->toString(), "b");
		cmd += Numpy::dot("b", "a", "res");

		Testing::submit("Linalg::dot(Vector<int>, Matrix<int>)", cmd, res->toString());

		auto result = dot(b, b);

		cmd += Numpy::createArray(b->toString(), "b");
		cmd += Numpy::dot("b", "b", "res");

		Testing::submit("Linalg::dot(Vector<int>, Vector<int>)", cmd, to_string(result));

		auto re = dot(c, a);

		cmd = Numpy::createArray(a->toString(), "a");
		cmd += Numpy::createArray(c->toString(), "b");
		cmd += Numpy::dot("b", "a", "res");

		Testing::submit("Linalg::dot(Matrix<int>, Vector<int>)", cmd, re->toString());
	}

	void testInner() {
		string cmd;

		auto a = arange(6, 6, 1)->flatten();
		auto b = (Vector<int>*)a->copy();
		auto c = inner(a, b);

		cmd = Numpy::createArray(a->toString(), "a");
		cmd += Numpy::createArray(b->toString(), "b");
		cmd += Numpy::inner("a", "b", "res");

		Testing::submit("Linalg::inner(Vector<int>, Vector<int>)", cmd, to_string(c));

		auto d = (Matrix<int>*)a->reshape(new Shape(2, 3));
		auto e = arange(3, 3, 1)->flatten();
		auto f = inner(d, e);

		cmd = Numpy::createArray(d->toString(), "a");
		cmd += Numpy::createArray(e->toString(), "b");
		cmd += Numpy::inner("a", "b", "res");

		Testing::submit("Linalg::inner(Vector<int>, Matrix<int>)", cmd, f->toString());

		auto g = inner(e, d);

		cmd = Numpy::createArray(d->toString(), "a");
		cmd += Numpy::createArray(e->toString(), "b");
		cmd += Numpy::inner("b", "a", "res");

		Testing::submit("Linalg::inner(Matrix<int>, Vector<int>)", cmd, g->toString());

		auto h = inner(d, d);

		cmd = Numpy::createArray(d->toString(), "a");
		cmd += Numpy::createArray(d->toString(), "b");
		cmd += Numpy::inner("a", "b", "res");

		Testing::submit("Linalg::inner(Matrix<int>, Vector<int>)", cmd, h->toString());
	}

	void testOuter() {
		string cmd;

		auto a = arange(6, 6, 1)->flatten();
		auto b = (Vector<int>*)a->copy();
		auto c = outer(a, b);

		cmd = Numpy::createArray(a->toString(), "a");
		cmd += Numpy::createArray(b->toString(), "b");
		cmd += Numpy::outer("a", "b", "res");

		Testing::submit("Linalg::outer(Vector<int>, Vector<int>)", cmd, c->toString());

		auto d = (Matrix<int>*)a->reshape(new Shape(2, 3));
		auto e = arange(3, 3, 1)->flatten();
		auto f = outer(d, e);

		cmd = Numpy::createArray(d->toString(), "a");
		cmd += Numpy::createArray(e->toString(), "b");
		cmd += Numpy::outer("a", "b", "res");

		Testing::submit("Linalg::outer(Vector<int>, Matrix<int>)", cmd, f->toString());

		auto g = outer(e, d);

		cmd = Numpy::createArray(d->toString(), "a");
		cmd += Numpy::createArray(e->toString(), "b");
		cmd += Numpy::outer("b", "a", "res");

		Testing::submit("Linalg::outer(Matrix<int>, Vector<int>)", cmd, g->toString());

		auto h = outer(d, d);

		cmd = Numpy::createArray(d->toString(), "a");
		cmd += Numpy::createArray(d->toString(), "b");
		cmd += Numpy::outer("a", "b", "res");

		Testing::submit("Linalg::outer(Matrix<int>, Vector<int>)", cmd, h->toString());
	}

	void testDet() {
		string cmd;

		auto mat = arange(16, 4, 4);
		auto res = det(mat);

		cmd = Numpy::createArray(mat->toString(), "mat");
		cmd += Numpy::det("mat", "res");

		Testing::submit("Linalg::det(Matrix<int>)", cmd, to_string(res));
	}

	void testTrace() {
		string cmd;

		auto a = arange(16, 4, 4);
		auto res = trace(a, 2);

		cmd = Numpy::createArray(a->toString(), "mat");
		cmd += Numpy::trace("mat", "2", "res");

		Testing::submit("Linalg::trace(Matrix<int>, int)", cmd, to_string(res));

		auto b = arange(15, 5, 3);
		res = trace(b);

		cmd = Numpy::createArray(b->toString(), "mat");
		cmd += Numpy::trace("mat", "0", "res");

		Testing::submit("Linalg::trace(Matrix<int>)", cmd, to_string(res));
	}

	void testSolve() {
		string cmd;

		auto a = new Matrix<double>({ 1, 2, 3, 5 }, 2, 2);
		auto b = new Vector<double>({ 1, 2 });
		auto res = solve(a, b);

		cmd = Numpy::createArray(a->toString(), "mat");
		cmd += Numpy::createArray(b->toString(), "vec");
		cmd += Numpy::solve("mat", "vec", "res");

		Testing::submit("Linalg::solve(Matrix<int>, Vector<int>)", cmd, res->toString());
	}

	void testNorm() {
		string cmd;

		auto p = arange<double>(9, 3, 3);
		auto res = norm(p);

		cmd = Numpy::createArray(p->toString(), "p");
		cmd += Numpy::roots("p", "res");

		Testing::submit("Linalg.roots(Vector<double>)", cmd, to_string(res));
	}

	void testEig() {
		string cmd;

		auto p = arange<double>(9, 3, 3);
		std::pair<Vector<double>*, Vector<double>**> res = eig(p);

		cmd = Numpy::createArray(p->toString(), "p");
		cmd += Numpy::eig("p", "res");
		cmd += "out = [" + res.first->toString() + ", [";
		for (int i = 0; i < 3; i++) {
			cmd += res.second[i]->toString();
		}
		cmd += "]\n";

		Testing::submit("Linalg.eig(Matrix<double>)", cmd, "out");
	}

	void testEigvals() {
		string cmd;

		auto p = arange<double>(9, 3, 3);
		Vector<double>* res = eigvals(p);

		cmd = Numpy::createArray(p->toString(), "p");
		cmd += Numpy::eigvals("p", "res");

		Testing::submit("Linalg.eigvals(Matrix<double>)", cmd, res->toString());
	}

	void testRoots() {
		string cmd;

		auto p = new Vector<double>({ 1, 1, -1 });
		auto res = roots(p);

		cmd = Numpy::createArray("[1, 1, -1]", "p");
		cmd += Numpy::roots("p", "res");

		Testing::submit("Linalg.roots(Vector<double>)", cmd, res->toString());
	}

	void testInv() {
		string cmd;

		auto p = new Matrix<double>({ 1, 2, 3, 4 }, 2, 2);
		auto res = inv(p);

		cmd = Numpy::createArray(p->toString(), "p");
		cmd += Numpy::inv("p", "res");

		Testing::submit("Linalg.inv(Matrix<double>)", cmd, res->toString());
	}

	void testConvolve() {
		string cmd;

		auto a = arange<double>(9, 3, 3)->flatten();
		auto b = arange<double>(16, 4, 4)->flatten();
		auto res = convolve(a, b);

		cmd = Numpy::createArray(a->toString(), "a");
		cmd += Numpy::createArray(b->toString(), "b");
		cmd += Numpy::convolve("a", "b", "res");

		Testing::submit("Linalg.convolve(Vector<int>, Vector<int>)", cmd, res->toString());
	}

	void testClip() {
		string cmd;

		auto vec = arange(50, 5, 10)->flatten();
		auto res = clip(vec, 5, 40);

		cmd = Numpy::createArray(vec->toString(), "vec");
		cmd += Numpy::clip("vec", "5", "40", "res");

		Testing::submit("Linalg.clip(Vector<int>, int, int)", cmd, res->toString());

		res = clip(vec, 20);

		cmd = Numpy::createArray(vec->toString(), "vec");
		cmd += Numpy::clip("vec", "-50", "20", "res");

		Testing::submit("Linalg.clip(Vector<int>, int)", cmd, res->toString());
	}
}