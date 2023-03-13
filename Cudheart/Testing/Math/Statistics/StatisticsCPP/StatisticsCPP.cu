#include "StatisticsCPP.cuh"

namespace Cudheart::Testing::Math::CPP::Statistics {
	using namespace Cudheart::CPP::Math::Statistics;

	void testStatistics() {
		testPtp();
		testPercentile();
		testQuantile();
		testMedian();
		testAverage();
		testMean();
		testStd();
		testVar();
		testCov();
		testCorrcoef();
		testHistogram();
		testHistogram2d();
		testBincount();
		testDigitize();
	}

	void testPtp() {
		string cmd;

		auto vec = new Vector<double>({ 4, 9, 2, 10, 6, 9, 7, 12 });
		auto res = ptp(vec);

		cmd = Numpy::createArray(vec->toString(), "vec");
		cmd += Numpy::ptp("vec", "res");

		Testing::submit("Statistics::ptp(Vector<double>)", cmd, to_string(res));
	}

	void testPercentile() {
		string cmd;

		auto vec = new Vector<double>({ 4, 9, 2, 10, 6, 9, 7, 12 });
		auto res = percentile(vec, 5);

		cmd = Numpy::createArray(vec->toString(), "vec");
		cmd += Numpy::percentile("vec", "5", "res");

		Testing::submit("Statistics::percentile(Vector<double>, double)", cmd, to_string(res));

		auto qs = new Vector<float>({ 5, 44, 99, 100, 0, 50, 69.420 });
		vec = new Vector<double>({ 10, 7, 4, 3, 2, 1 });
		auto b = percentile(vec, qs);

		cmd = Numpy::createArray(vec->toString(), "vec");
		cmd += Numpy::createArray(qs->toString(), "qs");
		cmd += Numpy::percentile("vec", "qs", "res");

		Testing::submit("Statistics::percentile(Vector<double>, Vector<float>)", cmd, b->toString());
	}

	void testQuantile() {
		string cmd;

		auto vec = new Vector<double>({ 4, 9, 2, 10, 6, 9, 7, 12 });
		auto res = quantile(vec, 0.05);

		cmd = Numpy::createArray(vec->toString(), "vec");
		cmd += Numpy::quantile("vec", "0.05", "res");

		Testing::submit("Statistics::quantile(Vector<double>, double)", cmd, to_string(res));

		auto qs = new Vector<float>({ 0.05, 0.44, 0.99, 0.1, 0, 0.5, 0.69420 });
		vec = new Vector<double>({ 10, 7, 4, 3, 2, 1 });
		auto b = quantile(vec, qs);

		cmd = Numpy::createArray(vec->toString(), "vec");
		cmd += Numpy::createArray(qs->toString(), "qs");
		cmd += Numpy::quantile("vec", "qs", "res");

		Testing::submit("Statistics::quantile(Vector<double>, Vector<float>)", cmd, b->toString());
	}

	void testMedian() {
		string cmd;

		auto vec = new Vector<double>({ 4, 9, 2, 10, 6, 9, 7, 12 });
		auto res = median(vec);

		cmd = Numpy::createArray(vec->toString(), "vec");
		cmd += Numpy::median("vec", "res");

		Testing::submit("Statistics::median(Vector<double>)", cmd, to_string(res));
	}

	void testAverage() {
		string cmd;

		auto a = VectorOps::arange(1, 11, 1);
		auto b = VectorOps::arange(10, 0, -1);
		auto res = average(a, b);

		cmd = Numpy::createArray(a->toString(), "a");
		cmd += Numpy::createArray(b->toString(), "b");
		cmd += Numpy::average("a", "b", "res");

		Testing::submit("Statistics::average(Vector<int>, Vector<int>)", cmd, to_string(res));

		double c = average(a);

		cmd = Numpy::createArray(a->toString(), "a");
		cmd += Numpy::average("a", "None", "res");

		Testing::submit("Statistics::average(Vector<int>)", cmd, to_string(c));
	}

	void testMean() {
		string cmd;
		auto a = VectorOps::arange(1, 11, 1);
		double c = mean(a);

		cmd = Numpy::createArray(a->toString(), "a");
		cmd += Numpy::mean("a", "res");

		Testing::submit("Statistics::mean(Vector<int>)", cmd, to_string(c));
	}

	void testStd() {
		string cmd;
		auto a = VectorOps::arange(1, 11, 1);
		double c = std(a);

		cmd = Numpy::createArray(a->toString(), "a");
		cmd += Numpy::std("a", "res");

		Testing::submit("Statistics::std(Vector<int>)", cmd, to_string(c));
	}

	void testVar() {
		string cmd;
		auto a = VectorOps::arange(1, 11, 1)->castTo<double>();
		double c = var(a);

		cmd = Numpy::createArray(a->toString(), "a");
		cmd += Numpy::var("a", "res");

		Testing::submit("Statistics::var(Vector<int>)", cmd, to_string(c));
	}

	void testCov() {
		string cmd;

		auto a = arange(9, 3, 3)->castTo<double>();
		auto b = cov(a, true);

		cmd = Numpy::createArray(a->toString(), "a");
		cmd += Numpy::cov("a", "True", "res");
		Testing::submit("Statistics::cov(Matrix<int>, true)", cmd, b->toString());
	}

	void testCorrcoef() {
	}

	void testHistogram() {
	}

	void testHistogram2d() {
	}

	void testBincount() {
	}

	void testDigitize() {
	}
}