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
using Cudheart::VectorOps::empty;
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

		NDArray<T>* sorted = Cudheart::Logic::sort(a);

		double temp = (q / 100) * sorted->getSize();
		int index;

		if (temp != (int)temp) {
			if ((temp - (int)temp) >= 0.5) {
				index = ceil(temp);
			}
			else {
				index = floor(temp);
			}

			return sorted->get(index);
		}
		else {
			index = (int)temp;

			T i = sorted->get(index - 1);
			T j = sorted->get(index);

			return (i + j) / 2;
		}

		return NULL;
	}

	template <typename T>
	inline NDArray<T>* percentile(NDArray<T>* a, NDArray<float>* q) {
		NDArray<T>* out = q->emptyLike();

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
		NDArray<T>* out = q->emptyLike();

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
		T sumWeights;
		T sumA;

		for (int i = 0; i < a->getSize(); i++) {
			sumWeights += weights->get(i);
			sumA += (a->get(i) * weights->get(i));
		}

		return sumA / sumWeights;
	}

	template <typename T>
	inline T average(NDArray<T>* a) {
		T sum;

		for (int i = 0; i < a->getSize(); i++) {
			sum += a->get(i);
		}

		return sum / a->getSize();
	}

	template <typename T>
	inline T mean(NDArray<T>* a) {
		return average<T>(a);
	}

	template <typename T>
	inline T std(NDArray<T>* a) {
		T sum = 0;
		T m = mean<T>(a);

		for (int i = 0; i < a->getSize(); i++) {
			sum += std::pow(a->get(i) - m, 2);
		}

		return std::sqrt(sum / a->getSize());
	}

	template <typename T>
	inline T var(NDArray<T>* a) {
		NDArray<T>* x = a->emptyLike();
		T meanA = mean(a);

		for (int i = 0; i < a->getSize(); i++) {
			x->set(i, std::pow(std::abs(a->get(i) - meanA), 2));
		}

		return mean(x);
	}

	template <typename T>
	inline Matrix<T>* cov(Matrix<T>* m) {
		T v = Cudheart::CPP::Math::sum(m);

		Matrix<T>* nm = (Matrix<T>*)m->emptyLike();

		for (int i = 0; i < nm->getSize(); i++) {
			nm->set(i, m->get(i) - 1);
		}

		Matrix<T>* dotProduct = Cudheart::CPP::Math::Linalg::dot(mn, m->transpose());
		T v2 = v / (std::pow(v, 2) - v);
		Matrix<T>* mat = Cudheart::MatrixOps::fullLike(dotProduct, v2);

		Matrix<T>* res = Cudheart::CPP::Math::multiply(mat, dotProduct);

		delete dotProduct, nm, mat, v2, v;

		return res;
	}

	template <typename T>
	inline Matrix<T>* cov(Matrix<T>* m, bool rowvar) {
		if (!rowvar) {
			m = m->transpose();
			auto temp = cov(m);
			delete m;
			return temp;
		}

		return cov(m);
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
	Vector<T>* histogram(NDArray<T>* a, Vector<T>* bins, T low, T high) {
		Vector<T>* out = bins->emptyLike();
		T* arr = new T[bins->getSize() + 2];

		arr[0] = low;
		arr[bins->getSize() + 1] = high;

		for (int i = 0; i < bins->getSize(); i++) {
			out->set(i, 0);
			arr[i] = bins->get(i + 1);
		}

		for (int i = 0; i < a->getSize(); i++) {
			T elem = a->get(i);
			for (int j = 1; j < bins->getSize() + 2; j++) {
				if (elem >= arr[j - 1] && elem < arr[j]) {
					out->set(j - 1, out->get(j - 1) + 1);
					break;
				}
			}
		}

		for (int i = 0; i < bins->getSize() + 2; i++) {
			delete arr[i];
		}

		delete[] arr;
	}

	template <typename T>
	Vector<T>* histogram(NDArray<T>* a, Vector<T>* bins) {
		return histogram(a, bins, Cudheart::Logic::minimum(a), Cudheart::Logic::maximum(a));
	}

	template <typename T>
	Vector<T>* histogram(NDArray<T>* a, T low, T high) {
		return histogram(a, bins, 10, low, high);
	}

	template <typename T>
	Vector<T>* histogram(NDArray<T>* a, int bins) {
		Vector<T>* bins = Cudheart::Math::linspace<T>(low, high, bins);
		return histogram(a, bins, Cudheart::Logic::minimum(a), Cudheart::Logic::maximum(a));
	}

	template <typename T>
	Vector<T>* histogram(NDArray<T>* a) {
		return histogram(a, 10, Cudheart::Logic::minimum(a), Cudheart::Logic::maximum(a));
	}

	template <typename T>
	Matrix<T>* histogram2d(Vector<T>* x, Vector<T>* y, Vector<T>* binX, Vector<T>* binY, T lowX, T highX, T lowY, T highY) {
		Vector<T>* xHist = histogram(x, binX, lowX, highX);
		Vector<T>* yHist = histogram(y, binY, lowY, highY);

		Matrix<T>* out = new Matrix<T>(x->getSize(), y->getSize());

		for (int i = 0; i < x->getSize(); i++) {
			for (int j = 0; j < y->getSize(); j++) {
				out->set(i, j, xHist->get(i) * yHist->get(j));
			}
		}

		return out;
	}

	template <typename T>
	Matrix<T>* histogram2d(Vector<T>* x, Vector<T>* y, Vector<T>* binX, Vector<T>* binY) {
		Vector<T>* xHist = histogram(x, binX);
		Vector<T>* yHist = histogram(y, binY);

		Matrix<T>* out = new Matrix<T>(x->getSize(), y->getSize());

		for (int i = 0; i < x->getSize(); i++) {
			for (int j = 0; j < y->getSize(); j++) {
				out->set(i, j, xHist->get(i) * yHist->get(j));
			}
		}

		return out;
	}

	template <typename T>
	Matrix<T>* histogram2d(Vector<T>* x, Vector<T>* y, T lowX, T highX, T lowY, T highY) {
		Vector<T>* xHist = histogram(x, lowX, highX);
		Vector<T>* yHist = histogram(y, lowY, highY);

		Matrix<T>* out = new Matrix<T>(x->getSize(), y->getSize());

		for (int i = 0; i < x->getSize(); i++) {
			for (int j = 0; j < y->getSize(); j++) {
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
	inline int digitize(T x, Vector<T>* bins, bool right = false, bool BinIncreasing = true) {
		if (right) {
			if (BinIncreasing) {
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
			if (BinIncreasing) {
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
	inline Vector<int> digitize(Vector<T>* x, Vector<T>* bins, bool right = false, bool BinIncreasing = true) {
		Vector<int>* out = (x->emptyLike)->castTo<int>();

		for (int i = 0; i < x->getSize(); i++) {
			out->set(i, digitize(x->get(i), bins, right, BinIncreasing));
		}

		return out;
	}
}