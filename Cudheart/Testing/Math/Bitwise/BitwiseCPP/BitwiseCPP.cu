#include "BitwiseCPP.cuh"

namespace Cudheart::Testing::Math::CPP::Bitwise {
	using namespace Cudheart::CPP::Math::Bitwise;

	void testBitwise() {
		testBitwiseAnd();
		testBitwiseOr();
		testBitwiseXor();
		testBitwiseLeftShift();
		testBitwiseRightShift();
		testBbitwiseNot();
	}

	void testBitwiseAnd() {
		string cmd;

		auto a = new Vector({ 15, 45, 22, 91, 37 });
		auto b = new Vector({ 31, 69, 420, 11, 13 });
		auto res = bitwiseAnd(a, b);

		cmd += Numpy::createArray(a->toString(), "a");
		cmd += Numpy::createArray(b->toString(), "b");
		cmd += Numpy::bitwiseAnd("a", "b", "res");

		Testing::submit("Bitwise::BitwiseAnd(Vector<int>, Vector<int>)", cmd, res->toString());
	}

	void testBitwiseOr() {
		string cmd;

		auto a = new Vector({ 15, 45, 22, 91, 37 });
		auto b = new Vector({ 31, 69, 420, 11, 13 });
		auto res = bitwiseOr(a, b);

		cmd += Numpy::createArray(a->toString(), "a");
		cmd += Numpy::createArray(b->toString(), "b");
		cmd += Numpy::bitwiseOr("a", "b", "res");

		Testing::submit("Bitwise::bitwiseOr(Vector<int>, Vector<int>)", cmd, res->toString());
	}

	void testBitwiseXor() {
		string cmd;

		auto a = new Vector({ 15, 45, 22, 91, 37 });
		auto b = new Vector({ 31, 69, 420, 11, 13 });
		auto res = bitwiseXor(a, b);

		cmd += Numpy::createArray(a->toString(), "a");
		cmd += Numpy::createArray(b->toString(), "b");
		cmd += Numpy::bitwiseXor("a", "b", "res");

		Testing::submit("Bitwise::bitwiseXor(Vector<int>, Vector<int>)", cmd, res->toString());
	}

	void testBitwiseLeftShift() {
		string cmd;

		auto a = new Vector({ 15, 45, 22, 91, 37 });
		auto b = new Vector({ 2, 3, 5, 4, 3 });
		auto res = bitwiseLeftShift(a, b);

		cmd += Numpy::createArray(a->toString(), "a");
		cmd += Numpy::createArray(b->toString(), "b");
		cmd += Numpy::bitwiseLeftShift("a", "b", "res");

		Testing::submit("Bitwise::bitwiseLeftShift(Vector<int>, Vector<int>)", cmd, res->toString());
	}

	void testBitwiseRightShift() {
		string cmd;

		auto a = new Vector({ 15, 45, 22, 91, 37 });
		auto b = new Vector({ 2, 3, 5, 4, 3 });
		auto res = bitwiseRightShift(a, b);

		cmd += Numpy::createArray(a->toString(), "a");
		cmd += Numpy::createArray(b->toString(), "b");
		cmd += Numpy::bitwiseRightShift("a", "b", "res");

		Testing::submit("Bitwise::bitwiseRightShift(Vector<int>, Vector<int>)", cmd, res->toString());
	}

	void testBbitwiseNot() {
		string cmd;

		auto a = new Vector({ 15, 45, 22, 91, 37 });
		auto res = bitwiseNot(a);

		cmd += Numpy::createArray(a->toString(), "a");
		cmd += Numpy::bitwiseNot("a", "res");

		Testing::submit("Bitwise::BitwiseAnd(Vector<int>, Vector<int>)", cmd, res->toString());
	}
}