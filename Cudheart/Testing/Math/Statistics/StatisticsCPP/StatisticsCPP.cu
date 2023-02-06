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

		auto vec = new Vector<double>({4, 9, 2, 10, 6, 9, 7, 12});
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
	}

	void testQuantile() {

	}

	void testMedian() {

	}

	void testAverage() {

	}

	void testMean() {

	}

	void testStd() {

	}

	void testVar() {

	}

	void testCov() {

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