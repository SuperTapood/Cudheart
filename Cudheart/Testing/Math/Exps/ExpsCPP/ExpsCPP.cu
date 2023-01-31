#include "ExpsCPP.cuh"


namespace Cudheart::Testing::Math::CPP::Exps {

	using namespace Cudheart::CPP::Math::Exp;

	void testExps() {
		testLn();
		testLoga2();
		testLogan();
		testLoga10();
		testExpo();
		testExpom1();
		testExpo2();
		testLogaddexp();
		testLogaddexp2();
	}

	void testLn() {
		string cmd;

		auto a = new Vector<double>({ 1, 2.71828, 15, 50, 666, 69, 420 });
		auto res = ln(a);

		cmd += Numpy::createArray("[1, 2.71828, 15, 50, 666, 69, 420]", "a");
		cmd += Numpy::ln("a", "res");

		Testing::submit("Exps::ln(Vector<double>)", cmd, res->toString());
	}

	void testLoga2() {
		string cmd;

		auto a = new Vector<double>({ 1, 2.71828, 15, 50, 666, 69, 420 });
		auto res = loga2(a);

		cmd += Numpy::createArray("[1, 2.71828, 15, 50, 666, 69, 420]", "a");
		cmd += Numpy::loga2("a", "res");

		Testing::submit("Exps::loga2(Vector<double>)", cmd, res->toString());
	}

	void testLogan() {
		string cmd;

		auto a = new Vector<double>({ 1, 2.71828, 15, 50, 666, 69, 420 });
		auto b = arange<double>(7, 7, 1)->flatten();
		auto res = logan(a, b);

		cmd = Numpy::createArray("[1, 2.71828, 15, 50, 666, 69, 420]", "a");
		cmd = Numpy::createArray(b->toString(), "b");
		cmd += Numpy::logan("a", "b", "res");

		Testing::submit("Exps::ln(Vector<double>, Vector<double>)", cmd, res->toString());

		res = logan(a, 4.0);

		cmd = Numpy::createArray("[1, 2.71828, 15, 50, 666, 69, 420]", "a");
		cmd += Numpy::logan("a", "4", "res");

		Testing::submit("Exps::ln(Vector<double>, double)", cmd, res->toString());
	}

	void testLoga10() {
		string cmd;

		auto a = new Vector<double>({ 1, 2.71828, 15, 50, 666, 69, 420 });
		auto res = loga10(a);

		cmd += Numpy::createArray("[1, 2.71828, 15, 50, 666, 69, 420]", "a");
		cmd += Numpy::loga10("a", "res");

		Testing::submit("Exps::loga10(Vector<double>)", cmd, res->toString());
	}

	void testExpo() {
		string cmd;

		auto a = new Vector<double>({ 1, 2.71828, 15, 50, 666, 69, 420 });
		auto res = expo(a);

		cmd += Numpy::createArray("[1, 2.71828, 15, 50, 666, 69, 420]", "a");
		cmd += Numpy::expo("a", "res");

		Testing::submit("Exps::expo(Vector<double>)", cmd, res->toString());
	}

	void testExpom1() {
		string cmd;

		auto a = new Vector<double>({ 1, 2.71828, 15, 50, 666, 69, 420 });
		auto res = expom1(a);

		cmd += Numpy::createArray("[1, 2.71828, 15, 50, 666, 69, 420]", "a");
		cmd += Numpy::expom1("a", "res");

		Testing::submit("Exps::expom1(Vector<double>)", cmd, res->toString());
	}

	void testExpo2() {
		string cmd;

		auto a = new Vector<double>({ 1, 2.71828, 15, 50, 666, 69, 420 });
		auto res = expo2(a);

		cmd += Numpy::createArray("[1, 2.71828, 15, 50, 666, 69, 420]", "a");
		cmd += Numpy::expo2("a", "res");

		Testing::submit("Exps::expo2(Vector<double>)", cmd, res->toString());
	}

	void testLogaddexp() {
		string cmd;

		auto a = new Vector<double>({ 1, 2.71828, 15, 50, 666, 69, 420 });
		auto b = arange<double>(7, 7, 1)->flatten();
		auto res = logaddexp(a, b);

		cmd = Numpy::createArray("[1, 2.71828, 15, 50, 666, 69, 420]", "a");
		cmd = Numpy::createArray(b->toString(), "b");
		cmd += Numpy::logaddexp("a", "b", "res");

		Testing::submit("Exps::logaddexp(Vector<double>, Vector<double>)", cmd, res->toString());
	}

	void testLogaddexp2() {
		string cmd;

		auto a = new Vector<double>({ 1, 2.71828, 15, 50, 666, 69, 420 });
		auto b = arange<double>(7, 7, 1)->flatten();
		auto res = logaddexp2(a, b);

		cmd = Numpy::createArray("[1, 2.71828, 15, 50, 666, 69, 420]", "a");
		cmd = Numpy::createArray(b->toString(), "b");
		cmd += Numpy::logaddexp2("a", "b", "res");

		Testing::submit("Exps::logaddexp2(Vector<double>, Vector<double>)", cmd, res->toString());
	}
}