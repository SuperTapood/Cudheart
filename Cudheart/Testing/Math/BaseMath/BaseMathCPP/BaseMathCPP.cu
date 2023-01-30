#include "BaseMathCPP.cuh"

namespace Cudheart::Testing::Math::CPP {
	using namespace Cudheart::CPP::Math::BaseMath;

	void testBaseMath() {
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
		string cmd;

		auto a = new Vector({ 14, 102, 55 });
		auto b = new Vector({ 10, 51, 11 });
		auto res = lcm(a, b);

		cmd = Numpy::createArray(a->toString(), "a");
		cmd += Numpy::createArray(b->toString(), "b");
		cmd += Numpy::lcm("a", "b", "res");

		Testing::submit("BaseMath::lcm(Vector<int>, Vector<int>)", cmd, res->toString());
	}

	void testGcd() {
		string cmd;

		auto a = new Vector({ 14, 102, 55 });
		auto b = new Vector({ 10, 51, 11 });
		auto res = gcd(a, b);

		cmd = Numpy::createArray(a->toString(), "a");
		cmd += Numpy::createArray(b->toString(), "b");
		cmd += Numpy::gcd("a", "b", "res");

		Testing::submit("BaseMath::gcd(Vector<int>, Vector<int>)", cmd, res->toString());
	}

	void testAdd() {
		string cmd;

		auto a = new Vector({ 1.584, 8.45475, 27.5, -54.9, -2.2 });
		auto b = new Vector<double>({ -1, -1, 1, -1, 1 });
		auto res = add(a, b);

		cmd = Numpy::createArray("[1.584, 8.45475, 27.5, -54.9, -2.2]", "a");
		cmd += Numpy::createArray("[-1, -1, 1, -1, 1]", "b");
		cmd += Numpy::add("a", "b", "res");

		Testing::submit("BaseMath::add(Vector<double>, Vector<double>)", cmd, res->toString());
	}

	void testSubtract() {
		string cmd;

		auto a = new Vector({ 1.584, 8.45475, 27.5, -54.9, -2.2 });
		auto b = new Vector<double>({ -1, -1, 1, -1, 1 });
		auto res = subtract(a, b);

		cmd = Numpy::createArray("[1.584, 8.45475, 27.5, -54.9, -2.2]", "a");
		cmd += Numpy::createArray("[-1, -1, 1, -1, 1]", "b");
		cmd += Numpy::subtract("a", "b", "res");

		Testing::submit("BaseMath::subtract(Vector<double>, Vector<double>)", cmd, res->toString());
	}

	void testMultiply() {
		string cmd;

		auto a = new Vector({ 1.584, 8.45475, 27.5, -54.9, -2.2 });
		auto b = new Vector<double>({ -1, -1, 1, -1, 1 });
		auto res = multiply(a, b);

		cmd = Numpy::createArray("[1.584, 8.45475, 27.5, -54.9, -2.2]", "a");
		cmd += Numpy::createArray("[-1, -1, 1, -1, 1]", "b");
		cmd += Numpy::multiply("a", "b", "res");

		Testing::submit("BaseMath::multiply(Vector<double>, Vector<double>)", cmd, res->toString());
	}

	void testDivide() {
		string cmd;

		auto a = new Vector({ 1.584, 8.45475, 27.5, -54.9, -2.2 });
		auto b = new Vector<double>({ -1, -1, 1, -1, 1 });
		auto res = divide(b, a);

		cmd = Numpy::createArray("[1.584, 8.45475, 27.5, -54.9, -2.2]", "a");
		cmd += Numpy::createArray("[-1, -1, 1, -1, 1]", "b");
		cmd += Numpy::divide("b", "a", "res");

		Testing::submit("BaseMath::divide(Vector<double>, Vector<double>)", cmd, res->toString());
	}

	void testFloorDivide() {
		string cmd;

		auto a = new Vector({ 1.584, 8.45475, 27.5, -54.9, -2.2 });
		auto b = new Vector<double>({ -1, -1, 1, -1, 1 });
		auto res = floorDivide(b, a);

		cmd = Numpy::createArray("[1.584, 8.45475, 27.5, -54.9, -2.2]", "a");
		cmd += Numpy::createArray("[-1, -1, 1, -1, 1]", "b");
		cmd += Numpy::floorDivide("b", "a", "res");

		Testing::submit("BaseMath::floorDivide(Vector<double>, Vector<double>)", cmd, res->toString());
	}

	void testMod() {
		string cmd;

		auto a = new Vector({ 14, 102, 55 });
		auto b = new Vector({ 10, 51, 11 });
		auto res = mod(a, b);

		cmd = Numpy::createArray(a->toString(), "a");
		cmd += Numpy::createArray(b->toString(), "b");
		cmd += Numpy::mod("a", "b", "res");

		Testing::submit("BaseMath::mod(Vector<int>, Vector<int>)", cmd, res->toString());
	}

	void testDivMod() {
		string cmd;

		auto a = new Vector({ 14, 102, 55 });
		auto b = new Vector({ 10, 51, 11 });
		auto res = divMod(a, b);

		cmd = Numpy::createArray(a->toString(), "a");
		cmd += Numpy::createArray(b->toString(), "b");
		cmd += Numpy::divMod("a", "b", "res");
		cmd += "res = np.array([" + res[0]->toString() + ", " + res[1]->toString() + "]\n)";

		Testing::submit("BaseMath::divMod(Vector<int>, Vector<int>)", cmd, "res");
	}

	void testReciprocal() {
		string cmd;

		auto a = new Vector({ 14, 102, 55 });
		auto res = reciprocal(a);

		cmd = Numpy::createArray(a->toString(), "a");
		cmd += Numpy::reciprocal("a", "res");

		Testing::submit("BaseMath::reciprocal(Vector<int>,)", cmd, res->toString());
	}

	void testPositive() {
		string cmd;

		auto vec = new Vector({ 1.584, 8.45475, 27.5, -54.9, -2.2 });
		auto res = positive(vec);

		cmd = Numpy::createArray("[1.584, 8.45475, 27.5, -54.9, -2.2]", "vec");
		cmd += Numpy::positive("vec", "res");

		Testing::submit("BaseMath::positive(Vector<double>)", cmd, res->toString());
	}

	void testNegative() {
		string cmd;

		auto vec = new Vector({ 1.584, 8.45475, 27.5, -54.9, -2.2 });
		auto res = negative(vec);

		cmd = Numpy::createArray("[1.584, 8.45475, 27.5, -54.9, -2.2]", "vec");
		cmd += Numpy::negative("vec", "res");

		Testing::submit("BaseMath::negative(Vector<double>)", cmd, res->toString());
	}

	void testSign() {
		string cmd;

		auto vec = new Vector({ 1.584, 8.45475, 27.5, -54.9, -2.2 });
		auto res = sign(vec);

		cmd = Numpy::createArray("[1.584, 8.45475, 27.5, -54.9, -2.2]", "vec");
		cmd += Numpy::sign("vec", "res");

		Testing::submit("BaseMath::sign(Vector<double>)", cmd, res->toString());
	}

	void testHeaviside() {
		string cmd;

		auto a = new Vector({ 1.584, 8.45475, 27.5, -54.9, -2.2 });
		auto b = new Vector<double>({ -1, 0, 1, -1, 0 });
		auto res = heaviside(b, a);

		cmd = Numpy::createArray("[1.584, 8.45475, 27.5, -54.9, -2.2]", "a");
		cmd += Numpy::createArray("[-1, 0, 1, -1, 0]", "b");
		cmd += Numpy::heaviside("b", "a", "res");

		Testing::submit("BaseMath::heaviside(Vector<double>, Vector<double>)", cmd, res->toString());
	}
}