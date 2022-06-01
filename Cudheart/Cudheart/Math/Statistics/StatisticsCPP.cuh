#pragma once

#include "../../Arrays/Arrays.cuh"
#include <cmath>
#include "../../Logic/Logic.cuh"

using Cudheart::NDArrays::Vector;
using Cudheart::NDArrays::Matrix;
using Cudheart::NDArrays::NDArray;
using namespace Cudheart::Exceptions;
using Cudheart::VectorOps::empty;
using Cudheart::VectorOps::emptyLike;
using Cudheart::MatrixOps::fromVector;

namespace Cudheart::Math::CPP::Statistics {
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
}