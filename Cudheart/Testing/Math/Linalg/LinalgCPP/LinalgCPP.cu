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

		Testing::submit("Linalg::dot(Vector<int>, Matrix<int>)", cmd, to_string(result));
	}

	void testInner() {

	}

	void testOuter() {

	}

	void testDet() {

	}

	void testTrace() {

	}

	void testSolve() {

	}

	void testInv() {

	}

	void testConvolve() {

	}

	void testClip() {

	}
}