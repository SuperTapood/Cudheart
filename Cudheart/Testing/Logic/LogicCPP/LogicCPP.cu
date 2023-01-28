#include "LogicCPP.cuh"


namespace Cudheart::Testing::Logic::CPP {
	using namespace Cudheart::Logic;
	void test() {
		testAll();
		testAny();
		testLogicalAnd();
		testLogicalOr();
		testLogicalNot();
		testLogicalXor();
		testAllclose();
		testEquals();
		testGreater();
		testGreaterEquals();
		testLess();
		testLessEqual();
		testMaximum();
		testAmax();
		testMinimum();
		testAmin();
	}

	void testAll() {
		string cmd;

		auto vec = new Vector({ true, true, false, true, true });
		auto a = all(vec);

		cmd += Numpy::createArray("[true, true, false, true, true]", "vec");
		cmd += Numpy::all("vec", "res");

		Testing::submit("Logic::all(Vector<bool>)", cmd, to_string(a));
	}

	void testAny() {
		string cmd;

		auto vec = new Vector({ true, true, false, true, true });
		auto a = any(vec);

		cmd += Numpy::createArray("[true, true, false, true, true]", "vec");
		cmd += Numpy::any("vec", "res");

		Testing::submit("Logic::any(Vector<bool>)", cmd, to_string(a));
	}

	void testLogicalAnd() {
		string cmd;

		auto a = new Vector({ true, true, false, true, true });
		auto b = new Vector({ false, true, true, false, true });
		auto res = logicalAnd(a, b);

		cmd += Numpy::createArray("[true, true, false, true, true]", "a");
		cmd += Numpy::createArray("[false, true, true, false, true]", "b");
		cmd += Numpy::logicalAnd("a", "b", "res");

		Testing::submit("Logic::logicalAnd(Vector<bool>)", cmd, res->toString());
	}

	void testLogicalOr() {
		string cmd;

		auto a = new Vector({ true, true, false, true, true });
		auto b = new Vector({ false, true, true, false, true });
		auto res = logicalOr(a, b);

		cmd += Numpy::createArray("[true, true, false, true, true]", "a");
		cmd += Numpy::createArray("[false, true, true, false, true]", "b");
		cmd += Numpy::logicalOr("a", "b", "res");

		Testing::submit("Logic::logicalOr(Vector<bool>)", cmd, res->toString());
	}

	void testLogicalNot() {
		string cmd;

		auto a = new Vector({ true, true, false, true, true });
		auto res = logicalNot(a);

		cmd += Numpy::createArray("[true, true, false, true, true]", "a");
		cmd += Numpy::logicalNot("a", "res");

		Testing::submit("Logic::logicalNot(Vector<bool>)", cmd, res->toString());
	}

	void testLogicalXor() {
		string cmd;

		auto a = new Vector({ true, true, false, true, true });
		auto b = new Vector({ false, true, true, false, true });
		auto res = logicalXor(a, b);

		cmd += Numpy::createArray("[true, true, false, true, true]", "a");
		cmd += Numpy::createArray("[false, true, true, false, true]", "b");
		cmd += Numpy::logicalXor("a", "b", "res");

		Testing::submit("Logic::logicalXor(Vector<bool>)", cmd, res->toString());
	}

	void testAllclose() {
		string cmd;

		auto a = new Vector({ 255, 278, 27872, 2, 278 });
		auto b = new Vector({ 2782, 278278, 2222, 2, 278 });
		auto res = allclose(a, b);

		cmd += Numpy::createArray(a->toString(), "a");
		cmd += Numpy::createArray(b->toString(), "b");
		cmd += Numpy::allclose("a", "b", "1e-05", "1e-08", "res");

		Testing::submit("Logic::allclose(Vector<int>)", cmd, to_string(res));
	}


	void testIsClose() {
		string cmd;

		auto a = new Vector({ 255, 278, 27872, 2, 278 });
		auto b = new Vector({ 2782, 278278, 2222, 2, 278 });
		auto res = isclose(a, b);

		cmd += Numpy::createArray(a->toString(), "a");
		cmd += Numpy::createArray(b->toString(), "b");
		cmd += Numpy::isclose("a", "b", "1e-05", "1e-08", "res");

		Testing::submit("Logic::isclose(Vector<int>)", cmd, res->toString());
	}


	void testEquals() {
		string cmd;

		auto a = new Vector({ true, true, false, true, true });
		auto b = new Vector({ false, true, true, false, true });
		auto res = equals(a, b);

		cmd += Numpy::createArray("[true, true, false, true, true]", "a");
		cmd += Numpy::createArray("[false, true, true, false, true]", "b");
		cmd += Numpy::equals("a", "b", "res");

		Testing::submit("Logic::equals(Vector<bool>)", cmd, res->toString());
	}

	void testGreater() {
		string cmd;

		auto a = new Vector({ 255, 278, 27872, 2, 278 });
		auto b = new Vector({ 2782, 278278, 2222, 2, 278 });
		auto res = greater(a, b);

		cmd = Numpy::createArray(a->toString(), "a");
		cmd += Numpy::createArray(b->toString(), "b");
		cmd += Numpy::greater("a", "b", "res");

		Testing::submit("Logic::greater(Vector<int>, Vector<int>)", cmd, res->toString());

		res = greater(a, b->get(2));

		cmd = Numpy::createArray(a->toString(), "a");
		cmd += Numpy::createArray(b->toString(), "b");
		cmd += Numpy::greater("a", "b[2]", "res");

		Testing::submit("Logic::greater(Vector<int>, int)", cmd, res->toString());
	}

	void testGreaterEquals() {
		string cmd;

		auto a = new Vector({ 255, 278, 27872, 2, 278 });
		auto b = new Vector({ 2782, 278278, 2222, 2, 278 });
		auto res = greaterEquals(a, b);

		cmd = Numpy::createArray(a->toString(), "a");
		cmd += Numpy::createArray(b->toString(), "b");
		cmd += Numpy::greaterEquals("a", "b", "res");

		Testing::submit("Logic::greaterEquals(Vector<int>, Vector<int>)", cmd, res->toString());

		res = greaterEquals(a, b->get(2));

		cmd = Numpy::createArray(a->toString(), "a");
		cmd += Numpy::createArray(b->toString(), "b");
		cmd += Numpy::greaterEquals("a", "b[2]", "res");

		Testing::submit("Logic::greaterEquals(Vector<int>, int)", cmd, res->toString());
	}

	void testLess() {
		string cmd;

		auto a = new Vector({ 255, 278, 27872, 2, 278 });
		auto b = new Vector({ 2782, 278278, 2222, 2, 278 });
		auto res = less(a, b);

		cmd = Numpy::createArray(a->toString(), "a");
		cmd += Numpy::createArray(b->toString(), "b");
		cmd += Numpy::less("a", "b", "res");

		Testing::submit("Logic::less(Vector<int>, Vector<int>)", cmd, res->toString());

		res = less(a, b->get(2));

		cmd = Numpy::createArray(a->toString(), "a");
		cmd += Numpy::createArray(b->toString(), "b");
		cmd += Numpy::less("a", "b[2]", "res");

		Testing::submit("Logic::less(Vector<int>, int)", cmd, res->toString());
	}

	void testLessEqual() {
		string cmd;

		auto a = new Vector({ 255, 278, 27872, 2, 278 });
		auto b = new Vector({ 2782, 278278, 2222, 2, 278 });
		auto res = lessEqual(a, b);

		cmd = Numpy::createArray(a->toString(), "a");
		cmd += Numpy::createArray(b->toString(), "b");
		cmd += Numpy::lessEqual("a", "b", "res");

		Testing::submit("Logic::equals(Vector<int>, Vector<int>)", cmd, res->toString());

		res = lessEqual(a, b->get(2));

		cmd = Numpy::createArray(a->toString(), "a");
		cmd += Numpy::createArray(b->toString(), "b");
		cmd += Numpy::lessEqual("a", "b[2]", "res");

		Testing::submit("Logic::equals(Vector<int>, int)", cmd, res->toString());
	}

	void testMaximum() {
		string cmd;

		auto a = new Vector({ 255, 278, 27872, 2, 278 });
		auto b = new Vector({ 2782, 278278, 2222, 2, 278 });
		auto res = maximum(a, b);

		cmd = Numpy::createArray(a->toString(), "a");
		cmd += Numpy::createArray(b->toString(), "b");
		cmd += Numpy::maximum("a", "b", "res");

		Testing::submit("Logic::maximum(Vector<int>, Vector<int>)", cmd, res->toString());
	}

	void testAmax() {
		string cmd;

		auto a = new Vector({ 255, 278, 27872, 2, 278 });
		auto res = amax(a);

		cmd = Numpy::createArray(a->toString(), "a");
		cmd += Numpy::amax("a", "res");

		Testing::submit("Logic::amax(Vector<int>, Vector<int>)", cmd, to_string(res));
	}

	void testMinimum() {
		string cmd;

		auto a = new Vector({ 255, 278, 27872, 2, 278 });
		auto b = new Vector({ 2782, 278278, 2222, 2, 278 });
		auto res = minimum(a, b);

		cmd = Numpy::createArray(a->toString(), "a");
		cmd += Numpy::createArray(b->toString(), "b");
		cmd += Numpy::minimum("a", "b", "res");

		Testing::submit("Logic::minimum(Vector<int>, Vector<int>)", cmd, res->toString());
	}

	void testAmin() {
		string cmd;

		auto a = new Vector({ 255, 278, 27872, 2, 278 });
		auto res = amin(a);

		cmd = Numpy::createArray(a->toString(), "a");
		cmd += Numpy::amin("a", "res");

		Testing::submit("Logic::amin(Vector<int>, Vector<int>)", cmd, to_string(res));
	}
}