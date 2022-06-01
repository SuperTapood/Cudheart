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
		a->assertMatchShape(weights);
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
		Matrix<T>* c = cov(x, rowvar);
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
}