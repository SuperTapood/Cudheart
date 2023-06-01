#include "ComplexMathCPP.cuh"

namespace Cudheart::Testing::Math::CPP::ComplexMath {
	using Cudheart::ComplexType;
	using namespace Cudheart::CPP::Math::Complex;

	void testComplexMath() {
		testAngle();
		testReal();
		testImag();
		testConj();
		testComplexAbs();
		testComplexSign();
	}

	void testAngle() {
		string cmd;

		auto a = new Vector<ComplexType*>(5);

		for (int i = 0; i < a->size(); i++) {
			a->set(i, new ComplexType(i + 1, i + 1));
		}

		auto res = angle(a, true);

		cmd = Numpy::createArray(a->toString(), "a");
		cmd += Numpy::angle("a", "true", "res");

		Testing::submit("Complex::angle(Vector<ComplexType*>, bool)", cmd, res->toString());

		res = angle(a);

		cmd = Numpy::createArray(a->toString(), "a");
		cmd += Numpy::angle("a", "false", "res");

		Testing::submit("Complex::angle(Vector<ComplexType*>)", cmd, res->toString());
	}

	void testReal() {
		string cmd;

		auto a = new Vector<ComplexType*>(5);

		for (int i = 0; i < a->size(); i++) {
			a->set(i, new ComplexType(i + 1, i + 1));
		}

		auto res = real(a);

		cmd = Numpy::createArray(a->toString(), "a");
		cmd += Numpy::real("a", "res");

		Testing::submit("Complex::real(Vector<ComplexType*>)", cmd, res->toString());
	}

	void testImag() {
		string cmd;

		auto a = new Vector<ComplexType*>(5);

		for (int i = 0; i < a->size(); i++) {
			a->set(i, new ComplexType(i + 1, i + 1));
		}

		auto res = imag(a);

		cmd = Numpy::createArray(a->toString(), "a");
		cmd += Numpy::imag("a", "res");

		Testing::submit("Complex::imag(Vector<ComplexType*>)", cmd, res->toString());
	}

	void testConj() {
		string cmd;

		auto a = new Vector<ComplexType*>(5);

		for (int i = 0; i < a->size(); i++) {
			a->set(i, new ComplexType(i + 1, i + 1));
		}

		auto res = conj(a);

		cmd = Numpy::createArray(a->toString(), "a");
		cmd += Numpy::conj("a", "res");

		Testing::submit("Complex::conj(Vector<ComplexType*>)", cmd, res->toString());
	}

	void testComplexAbs() {
		string cmd;

		auto a = new Vector<ComplexType*>(5);

		for (int i = 0; i < a->size(); i++) {
			a->set(i, new ComplexType(i + 1, i + 1));
		}

		auto res = complexAbs(a);

		cmd = Numpy::createArray(a->toString(), "a");
		cmd += Numpy::abs("a", "res");

		Testing::submit("Complex::complexAbs(Vector<ComplexType*>)", cmd, res->toString());
	}

	void testComplexSign() {
		string cmd;

		auto a = new Vector<ComplexType*>(5);

		for (int i = 0; i < a->size(); i++) {
			a->set(i, new ComplexType(i + 1, i + 1));
		}

		auto res = complexSign(a);

		cmd = Numpy::createArray(a->toString(), "a");
		cmd += Numpy::sign("a", "res");

		Testing::submit("Complex::complexSign(Vector<ComplexType*>)", cmd, res->toString());
	}
}