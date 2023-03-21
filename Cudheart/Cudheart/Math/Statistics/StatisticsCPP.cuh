#pragma once

#include "../../Arrays/Arrays.cuh"
#include <cmath>
#include "../../Logic/Logic.cuh"
#include "../BaseMath/BaseMath.cuh"
#include "../Linalg/Linalg.cuh"

using Cudheart::NDArrays::Vector;
using Cudheart::NDArrays::Matrix;
using Cudheart::NDArrays::NDArray;
using namespace Cudheart::Exceptions;
using Cudheart::VectorOps::emptyLike;
using Cudheart::MatrixOps::fromVector;

namespace Cudheart::CPP::Math::Statistics {
	template <typename T>
	int ptp(NDArray<T>* a) {
		return Cudheart::Logic::amax(a) - Cudheart::Logic::amin(a);
	}

	template <typename T>
	inline T percentile(NDArray<T>* a, float q) {
		// 0 <= q <= 100

		NDArray<T>* sorted = Cudheart::Sorting::sort(a);

		int size = a->getSize();
		// linear method
		double virt = ((q / 100) * (size - 1));

		T i = sorted->get(floor(virt));
		T j = sorted->get(ceil(virt));
		T dist = j - i;

		return i + dist * (virt - (int)virt);
	}

	template <typename T>
	inline NDArray<T>* percentile(NDArray<T>* a, NDArray<float>* q) {
		NDArray<T>* out = q->emptyLike<T>();

		for (int i = 0; i < q->getSize(); i++) {
			out->set(i, percentile(a, q->get(i)));
		}

		return out;
	}

	template <typename T>
	inline T quantile(NDArray<T>* a, float q) {
		return percentile(a, q * 100);
	}

	template <typename T>
	inline NDArray<T>* quantile(NDArray<T>* a, NDArray<float>* q) {
		NDArray<T>* out = q->emptyLike<T>();

		for (int i = 0; i < q->getSize(); i++) {
			out->set(i, quantile(a, q->get(i)));
		}

		return out;
	}

	template <typename T>
	inline T median(NDArray<T>* a) {
		return quantile(a, 0.5);
	}

	template <typename T>
	inline T average(NDArray<T>* a, NDArray<T>* weights) {
		a->assertMatchShape(weights->getShape());
		T sumWeights = 0;
		T sumA = 0;

		for (int i = 0; i < a->getSize(); i++) {
			sumWeights += weights->get(i);
			sumA += (a->get(i) * weights->get(i));
		}

		return sumA / sumWeights;
	}

	template <typename T>
	inline double average(NDArray<T>* a) {
		T sum = 0;

		for (int i = 0; i < a->getSize(); i++) {
			sum += a->get(i);
		}

		return (double)sum / (double)a->getSize();
	}

	template <typename T>
	inline double mean(NDArray<T>* a) {
		return (double)average<T>(a);
	}

	template <typename T>
	inline double std(NDArray<T>* a) {
		double sum = 0;
		double m = mean<T>(a);

		for (int i = 0; i < a->getSize(); i++) {
			sum += std::pow(a->get(i) - m, 2);
		}

		return std::sqrt(sum / a->getSize());
	}

	template <typename T>
	inline double var(NDArray<T>* a) {
		NDArray<double>* x = a->emptyLike<double>();
		T meanA = mean(a);

		for (int i = 0; i < a->getSize(); i++) {
			x->set(i, std::pow(std::abs(a->get(i) - meanA), 2));
		}

		return mean(x);
	}

	template <typename T>
	inline Matrix<T>* cov(Matrix<T>* m, bool rowvar = true) {
		if (!rowvar) {
			auto mat = m->transpose();
			auto temp = cov((Matrix<T>*)mat);
			delete mat;
			return temp;
		}
		T fact = m->getShape()->getY() - 1;
		Vector<T>* avg = new Vector<T>(m->getWidth());
		Matrix<T>* trans = (Matrix<T>*)m->transpose();
		for (int i = 0; i < avg->getSize(); i++) {
			avg->set(i, average(m->getRow(i)));
		}
		Matrix<T>* av = new Matrix<T>(avg->getSize(), avg->getSize());
		for (int i = 0; i < avg->getSize(); i++) {
			for (int j = 0; j < avg->getSize(); j++) {
				av->set(i, j, avg->get(i));
			}
		}
		Matrix<T>* X = (Matrix<T>*)BaseMath::subtract(m, av);
		Matrix<T>* X_T = (Matrix<T>*)X->transpose();
		Matrix<T>* c = (Matrix<T>*)Linalg::dot(X, X_T);
		Matrix<T>* res = (Matrix<T>*)BaseMath::multiply(c, fullLike(c, (T)(1 / fact)));
		return res;
	}

	template <typename T>
	Matrix<T>* corrcoef(Matrix<T>* x, bool rowvar = true) {
		Matrix<T>* c = cov<T>(x, rowvar);
		Matrix<T>* r = (Matrix<T>*)c->emptyLike();

		for (int i = 0; i < r->getHeight(); i++) {
			for (int j = 0; j < r->getWidth(); j++) {
				T high = c->get(i, j);
				T low = std::sqrt(c->get(i, i) * c->get(j, j));
				r->set(i, j, high / low);
			}
		}

		delete c;

		return r;
	}

	template <typename T>
	Vector<T>* histogram(NDArray<T>* a, Vector<T>* bins) {
		bins = (Vector<T>*)Sorting::sort(bins);

		Vector<T>* out = new Vector<T>(bins->getSize() - 1);

		for (int i = 0; i < out->getSize(); i++) {
			out->set(i, (T)0);
		}

		for (int i = 0; i < a->getSize(); i++) {
			T v = a->get(i);
			for (int j = 1; j < bins->getSize(); j++) {
				if (v < bins->get(j)) {
					out->set(j - 1, out->get(j - 1) + 1);
					break;
				}
			}
			if (bins->get(-1) == v) {
				out->set(-1, out->get(-1) + 1);
			}
		}

		return out;
	}

	template <typename T>
	Vector<T>* histogram(NDArray<T>* a, int bins) {
		T low = Logic::amin(a);
		T high = Logic::amax(a);
		Vector<T>* binnes = Cudheart::VectorOps::linspace<T>(low, high, bins);
		return histogram(a, binnes);
	}

	template <typename T>
	Vector<T>* histogram(NDArray<T>* a) {
		return histogram(a, 11);
	}

	template <typename T>
	Matrix<T>* histogram2d(Vector<T>* x, Vector<T>* y, Vector<T>* binX, Vector<T>* binY) {
		Vector<T>* xHist = histogram(x, binX);
		Vector<T>* yHist = histogram(y, binY);

		Matrix<T>* out = new Matrix<T>(xHist->getSize(), yHist->getSize());

		for (int i = 0; i < xHist->getSize(); i++) {
			for (int j = 0; j < yHist->getSize(); j++) {
				out->set(i, j, xHist->get(i) * yHist->get(j));
			}
		}

		return out;
	}

	template <typename T>
	Matrix<T>* histogram2d(Vector<T>* x, Vector<T>* y) {
		Vector<T>* xHist = histogram(x);
		Vector<T>* yHist = histogram(y);

		Matrix<T>* out = new Matrix<T>(xHist->getSize(), yHist->getSize());

		for (int i = 0; i < xHist->getSize(); i++) {
			for (int j = 0; j < yHist->getSize(); j++) {
				out->set(i, j, xHist->get(i) * yHist->get(j));
			}
		}

		return out;
	}

	template <typename T>
	inline Vector<int>* bincount(NDArray<T>* x) {
		// assert bins > 0
		// assert T is whole and maybe cast it
		// maybe this helps?
		Vector<int>* bins = new Vector<T>(Cudheart::Logic::maximum(x) + 1);

		for (int i = 0; i < bins->getSize(); i++) {
			int count = 0;
			for (int j = 0; j < x->getSize(); j++) {
				if (x->get(j) == i) {
					count++;
				}
			}
			bins->set(i, count);
		}

		return bins;
	}

	template <typename T>
	inline int digitize(T x, Vector<T>* bins, bool right = false) {
		bool binIncreasing = true;

		T value = bins->get(0);
		for (int i = 1; i < bins->getSize(); i++) {
			if (bins->get(i) < value) {
				binIncreasing = false;
				break;
			}
			value = bins->get(i);
		}

		if (!binIncreasing) {
			T value = bins->get(0);
			for (int i = 1; i < bins->getSize(); i++) {
				if (bins->get(i) > value) {
					Exceptions::BaseException("ValueError: bins have to be sorted").raise();
				}
				value = bins->get(i);
			}
		}

		if (right) {
			if (binIncreasing) {
				for (int i = 1; i < bins->getSize(); i++) {
					if (bins->get(i - 1) < x && x <= bins[i]) {
						return i;
					}
				}
			}
			else {
				for (int i = 1; i < bins->getSize(); i++) {
					if (bins->get(i - 1) >= x && x > bins[i]) {
						return i;
					}
				}
			}
		}
		else {
			if (binIncreasing) {
				for (int i = 1; i < bins->getSize(); i++) {
					if (bins->get(i - 1) <= x && x < bins[i]) {
						return i;
					}
				}
			}
			else {
				for (int i = 1; i < bins->getSize(); i++) {
					if (bins->get(i - 1) > x && x >= bins[i]) {
						return i;
					}
				}
			}
		}

		return -1;
	}

	template <typename T>
	inline Vector<int>* digitize(Vector<T>* x, Vector<T>* bins, bool right = false) {
		Vector<int>* out = new Vector(x->getSize());

		for (int i = 0; i < x->getSize(); i++) {
			out->set(i, digitize(x->get(i), bins, right));
		}

		return out;
	}
}