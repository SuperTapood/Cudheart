#include "BaseMathCPP.cuh"

namespace Cudheart::Testing::Math {
	using namespace Cudheart::CPP::BaseMath;

	void testBaseMathCPP() {
		testCubeRoot();
		testSquare();
		testSquareRoot();
		testPower();
		testAround();
		testRint();
		testFix();
		testFloor();
		testTrunc();
		testCeil();
		testProd();
		testSum();
		testCumProd();
		testCumSum();
		testSignBit();
		testCopySign();
		testAbs();
		testLcm();
		testGcd();
		testAdd();
		testSubtract();
		testMultiply();
		testDivide();
		testFloorDivide();
		testMod();
		testDivMod();
		testReciprocal();
		testPositive();
		testNegative();
		testSign();
		testHeaviside();
	}

	void testCubeRoot() {
		string cmd;

		auto vec = new Vector({ 1, 8, 27 });
		auto res = cubeRoot(vec);

		cmd += Numpy::createArray("[1, 8, 27]", "vec");
		cmd += Numpy::cubeRoot("vec", "res");

		Testing::submit("BaseMath::cubeRoot(Vector<int>)", cmd, res->toString());
	}

	void testSquare() {
		string cmd;

		auto vec = new Vector({ 1, 8, 27 });
		auto res = square(vec);

		cmd += Numpy::createArray("[1, 8, 27]", "vec");
		cmd += Numpy::square("vec", "res");

		Testing::submit("BaseMath::square(Vector<int>)", cmd, res->toString());
	}

	void testSquareRoot() {
		string cmd;

		auto vec = new Vector<double>({ 1, 8, 27 });
		auto res = squareRoot(vec);

		cmd += Numpy::createArray("[1, 8, 27]", "vec");
		cmd += Numpy::squareRoot("vec", "res");

		Testing::submit("BaseMath::squareRoot(Vector<double>)", cmd, res->toString());
	}

	void testPower() {
		string cmd;

		auto base = new Vector<double>({ 1, 8, 27 });
		auto pow = new Vector<double>({ 2, 6, 5 });
		auto res = power(base, pow);

		cmd += Numpy::createArray("[1, 8, 27]", "base");
		cmd += Numpy::createArray("[2, 6, 5]", "power");
		cmd += Numpy::power("base", "power", "res");

		Testing::submit("BaseMath::power(Vector<double>, Vector<double>)", cmd, res->toString());
	}

	void testAround() {
		string cmd;

		auto vec = new Vector({ 1.584, 8.45475, 27.5 });
		auto res = around(vec, 3);

		cmd = Numpy::createArray("[1.584, 8.45475, 27.5]", "vec");
		cmd += Numpy::around("vec", "3", "res");

		Testing::submit("BaseMath::around(Vector<double>, int)", cmd, res->toString());

		res = around(vec);

		cmd = Numpy::createArray("[1.584, 8.45475, 27.5]", "vec");
		cmd += Numpy::around("vec", "0", "res");

		Testing::submit("BaseMath::around(Vector<double>)", cmd, res->toString());
	}

	void testRint() {
		string cmd;

		auto vec = new Vector({ 1.584, 8.45475, 27.5 });
		auto res = rint(vec);

		cmd = Numpy::createArray("[1.584, 8.45475, 27.5]", "vec");
		cmd += Numpy::rint("vec", "res");

		Testing::submit("BaseMath::rint(Vector<double>)", cmd, res->toString());
	}

	void testFix() {
		string cmd;

		auto vec = new Vector({1.584, 8.45475, 27.5, -54.9, -2.2});
		auto res = fix(vec);

		cmd = Numpy::createArray("[1.584, 8.45475, 27.5, -54.9, -2.2]", "vec");
		cmd += Numpy::fix("vec", "res");

		Testing::submit("BaseMath::fix(Vector<double>)", cmd, res->toString());
	}

	void testFloor() {
		string cmd;

		auto vec = new Vector({ 1.584, 8.45475, 27.5, -54.9, -2.2 });
		auto res = floor(vec);

		cmd = Numpy::createArray("[1.584, 8.45475, 27.5, -54.9, -2.2]", "vec");
		cmd += Numpy::floor("vec", "res");

		Testing::submit("BaseMath::floor(Vector<double>)", cmd, res->toString());
	}

	void testTrunc() {
		string cmd;

		auto vec = new Vector({ 1.584, 8.45475, 27.5, -54.9, -2.2 });
		auto res = trunc(vec);

		cmd = Numpy::createArray("[1.584, 8.45475, 27.5, -54.9, -2.2]", "vec");
		cmd += Numpy::trunc("vec", "res");

		Testing::submit("BaseMath::trunc(Vector<double>)", cmd, res->toString());
	}

	void testCeil() {
		string cmd;

		auto vec = new Vector({ 1.584, 8.45475, 27.5, -54.9, -2.2 });
		auto res = ceil(vec);

		cmd = Numpy::createArray("[1.584, 8.45475, 27.5, -54.9, -2.2]", "vec");
		cmd += Numpy::ceil("vec", "res");

		Testing::submit("BaseMath::ceil(Vector<double>)", cmd, res->toString());
	}

	void testProd() {
		string cmd;

		auto vec = new Vector({ 1.584, 8.45475, 27.5, -54.9, -2.2 });
		auto res = prod(vec);

		cmd = Numpy::createArray("[1.584, 8.45475, 27.5, -54.9, -2.2]", "vec");
		cmd += Numpy::prod("vec", "res");

		Testing::submit("BaseMath::prod(Vector<double>)", cmd, to_string(res));
	}

	void testSum() {
		string cmd;

		auto vec = new Vector({ 1.584, 8.45475, 27.5, -54.9, -2.2 });
		auto res = sum(vec);

		cmd = Numpy::createArray("[1.584, 8.45475, 27.5, -54.9, -2.2]", "vec");
		cmd += Numpy::sum("vec", "res");

		Testing::submit("BaseMath::sum(Vector<double>)", cmd, to_string(res));
	}

	void testCumProd() {
		string cmd;

		auto vec = new Vector({ 1.584, 8.45475, 27.5, -54.9, -2.2 });
		auto res = cumProd(vec);

		cmd = Numpy::createArray("[1.584, 8.45475, 27.5, -54.9, -2.2]", "vec");
		cmd += Numpy::cumProd("vec", "res");

		Testing::submit("BaseMath::cumProd(Vector<double>)", cmd, res->toString());
	}

	void testCumSum() {
		string cmd;

		auto vec = new Vector({ 1.584, 8.45475, 27.5, -54.9, -2.2 });
		auto res = cumSum(vec);

		cmd = Numpy::createArray("[1.584, 8.45475, 27.5, -54.9, -2.2]", "vec");
		cmd += Numpy::cumSum("vec", "res");

		Testing::submit("BaseMath::cumSum(Vector<double>)", cmd, res->toString());
	}

	void testSignBit() {
		string cmd;

		auto vec = new Vector({ 1.584, 8.45475, 27.5, -54.9, -2.2 });
		auto res = signBit(vec);

		cmd = Numpy::createArray("[1.584, 8.45475, 27.5, -54.9, -2.2]", "vec");
		cmd += Numpy::signBit("vec", "res");

		Testing::submit("BaseMath::signBit(Vector<double>)", cmd, res->toString());
	}

	void testCopySign() {
		string cmd;

		auto a = new Vector({ 1.584, 8.45475, 27.5, -54.9, -2.2 });
		auto b = new Vector<double>({ -1, -1, 1, -1, 1 });
		auto res = copySign(a, b);

		cmd = Numpy::createArray("[1.584, 8.45475, 27.5, -54.9, -2.2]", "a");
		cmd += Numpy::createArray("[-1, -1, 1, -1, 1]", "b");
		cmd += Numpy::copySign("a", "b", "res");

		Testing::submit("BaseMath::copySign(Vector<double>, Vector<double>)", cmd, res->toString());
	}

	void testAbs() {
		string cmd;

		auto vec = new Vector({ 1.584, 8.45475, 27.5, -54.9, -2.2 });
		auto res = abs(vec);

		cmd = Numpy::createArray("[1.584, 8.45475, 27.5, -54.9, -2.2]", "vec");
		cmd += Numpy::abs("vec", "res");

		Testing::submit("BaseMath::abs(Vector<double>)", cmd, res->toString());
	}

	void testLcm() {

	}

	void testGcd() {

	}

	void testAdd() {

	}

	void testSubtract() {

	}

	void testMultiply() {

	}

	void testDivide() {

	}

	void testFloorDivide() {

	}

	void testMod() {

	}

	void testDivMod() {

	}

	void testReciprocal() {

	}

	void testPositive() {

	}

	void testNegative() {

	}

	void testSign() {

	}

	void testHeaviside() {

	}
}