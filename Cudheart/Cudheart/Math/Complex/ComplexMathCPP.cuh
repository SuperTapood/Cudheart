#pragma once

#include "../../Arrays/Arrays.cuh"
#include <cmath>
#include <numeric>
#include "ComplexType.cuh"
#include "../../Constants.cuh"

using Cudheart::NDArrays::NDArray;
using Cudheart::NDArrays::Vector;
using Cudheart::NDArrays::Matrix;
using namespace Cudheart::Exceptions;
using Cudheart::VectorOps::emptyLike;
using Cudheart::MatrixOps::fromVector;
using Cudheart::Constants::pi;

namespace Cudheart::CPP::Math::Complex {
	inline NDArray<double>* angle(NDArray<ComplexType*>* z, bool deg) {
		auto arr = (new Vector<double>(z->size()))->reshape(z->getShape());
		double multiplier = 1.0;

		if (deg) {
			multiplier = 180 / pi;
		}

		for (int i = 0; i < z->size(); i++) {
			auto current = z->get(i);
			arr->set(i, std::atan(current->real / current->real) * multiplier);
		}

		return arr;
	}

	inline NDArray<double>* angle(NDArray<ComplexType*>* z) {
		return angle(z, false);
	}

	inline NDArray<double>* real(NDArray<ComplexType*>* val) {
		NDArray<double>* arr = (new Vector<double>(val->size()))->reshape(val->getShape());

		for (int i = 0; i < val->size(); i++) {
			ComplexType* current = val->get(i);
			arr->set(i, current->real);
		}

		return arr;
	}

	inline NDArray<double>* imag(NDArray<ComplexType*>* val) {
		NDArray<double>* arr = (new Vector<double>(val->size()))->reshape(val->getShape());

		for (int i = 0; i < val->size(); i++) {
			ComplexType* current = val->get(i);
			arr->set(i, current->real);
		}

		return arr;
	}

	inline NDArray<ComplexType*>* conj(NDArray<ComplexType*>* x) {
		NDArray<ComplexType*>* arr = x->emptyLike();

		for (int i = 0; i < x->size(); i++) {
			ComplexType* current = x->get(i);
			arr->set(i, new ComplexType(current->real, -current->imag));
		}

		return arr;
	}

	inline NDArray<double>* complexAbs(NDArray<ComplexType*>* x) {
		NDArray<double>* out = (new Vector<double>(x->size()))->reshape(x->getShape());

		for (int i = 0; i < x->size(); i++) {
			ComplexType* current = x->get(i);
			double x2 = std::pow(current->real, 2);
			double y2 = std::pow(current->imag, 2);
			out->set(i, std::sqrt(x2 + y2));
		}

		return out;
	}

	inline NDArray<ComplexType*>* complexSign(NDArray<ComplexType*>* x) {
		NDArray<ComplexType*>* out = x->emptyLike();

		for (int i = 0; i < x->size(); i++) {
			double real = x->get(i)->real;
			if (std::signbit(real)) {
				out->set(i, new ComplexType(-1, 0));
			}
			else {
				out->set(i, new ComplexType(1, 0));
			}
		}

		return out;
	}
}