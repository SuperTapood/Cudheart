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
		auto b = new Vector<double>({1, 2});
		auto res = solve(a, b);

		cmd = Numpy::createArray(a->toString(), "mat");
		cmd += Numpy::createArray(b->toString(), "vec");
		cmd += Numpy::solve("mat", "vec", "res");

		Testing::submit("Linalg::solve(Matrix<int>, Vector<int>)", cmd, res->toString());
	}

	void testInv() {

	}

	void testConvolve() {

	}

	void testClip() {

	}
}