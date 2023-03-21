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

		auto d = cov(a, false);

		cmd = Numpy::createArray(a->toString(), "a");
		cmd += Numpy::cov("a", "False", "res");
		Testing::submit("Statistics::cov(Matrix<int>, false)", cmd, d->toString());

		auto f = cov(a);

		cmd = Numpy::createArray(a->toString(), "a");
		cmd += Numpy::cov("a", "True", "res");
		Testing::submit("Statistics::cov(Matrix<int>)", cmd, f->toString());
	}

	void testCorrcoef() {
		string cmd;

		auto a = arange(9, 3, 3)->castTo<double>();
		auto b = corrcoef(a, true);

		cmd = Numpy::createArray(a->toString(), "a");
		cmd += Numpy::corrcoef("a", "True", "res");
		Testing::submit("Statistics::corrcoef(Matrix<int>, true)", cmd, b->toString());

		auto d = corrcoef(a, false);

		cmd = Numpy::createArray(a->toString(), "a");
		cmd += Numpy::corrcoef("a", "False", "res");
		Testing::submit("Statistics::corrcoef(Matrix<int>, false)", cmd, d->toString());

		auto f = corrcoef(a);

		cmd = Numpy::createArray(a->toString(), "a");
		cmd += Numpy::corrcoef("a", "True", "res");
		Testing::submit("Statistics::corrcoef(Matrix<int>)", cmd, f->toString());
	}

	void testHistogram() {
		string cmd;

		auto a = new Vector({1.0, 2.0, 1.0 });
		auto b = new Vector({ 0.0, 1.0, 2.0, 3.0});

		auto c = histogram(a, b);
		
		cmd = Numpy::createArray(a->toString(), "a");
		cmd += Numpy::createArray(b->toString(), "b");
		cmd += "res, _ = np.histogram(a, bins = b)";
		Testing::submit("Statistics::histogram(Vector<int>, Vector<int>)", cmd, c->toString());

		c = histogram(a);

		cmd = Numpy::createArray(a->toString(), "a");
		cmd += "res, _ = np.histogram(a)";
		Testing::submit("Statistics::histogram(Vector<int>)", cmd, c->toString());
	}

	void testHistogram2d() {
		string cmd;

		auto a = new Vector({ 1.0, 2.0, 1.0 });
		auto aa = new Vector({ 5.0, 7.0, 3.0 });
		auto b = new Vector({ 0.0, 1.0, 2.0, 3.0 });
		auto bb = new Vector({ 3.0, 6.0, 9.0, 10.0 });

		auto c = histogram2d(a, aa, b, bb);

		cmd = Numpy::createArray(a->toString(), "a");
		cmd += Numpy::createArray(b->toString(), "b");
		cmd = Numpy::createArray(aa->toString(), "aa");
		cmd += Numpy::createArray(bb->toString(), "bb");
		cmd += "res, _, _ = np.histogram2d(a, aa, bins=(b, bb))";
		Testing::submit("Statistics::histogram2d(Vector<int>, Vector<int>, Vector<int>, Vector<int>)", cmd, c->toString());

		c = histogram2d(a, aa);

		cmd = Numpy::createArray(a->toString(), "a");
		cmd = Numpy::createArray(aa->toString(), "aa");
		cmd += "res, _, _ = np.histogram2d(a, aa)";
		Testing::submit("Statistics::histogram2d(Vector<int>, Vector<int>)", cmd, c->toString());
	}

	void testBincount() {
	}

	void testDigitize() {
	}
}