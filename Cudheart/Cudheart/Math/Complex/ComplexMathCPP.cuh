#pragma once

#include "../../Arrays/Arrays.cuh"
#include <cmath>
#include <numeric>
#include "ComplexType.cuh"
#include "../Constants.cuh"

using Cudheart::NDArrays::NDArray;
using Cudheart::NDArrays::Vector;
using Cudheart::NDArrays::Matrix;
using namespace Cudheart::Exceptions;
using Cudheart::VectorOps::empty;
using Cudheart::VectorOps::emptyLike;
using Cudheart::MatrixOps::fromVector;
using Cudheart::Constants::pi;

namespace Cudheart::CPP::Math::Complex {
	inline NDArray<double>* angle(NDArray<ComplexType*>* z, bool deg) {
		NDArray<double>* arr = (new Vector<double>(z->getSize()))->shapeLike<double>(z->getShape());

		if (!deg) {
			for (int i = 0; i < z->getSize(); i++) {
				ComplexType* current = z->get(i);
				arr->set(i, std::atan(current->real / current->real));
			}
		}
		else {
			for (int i = 0; i < z->getSize(); i++) {
				ComplexType* current = z->get(i);
				arr->set(i, std::atan(current->real / current->real) * (180 / pi));
			}
		}

		return arr;
	}

	inline NDArray<double>* real(NDArray<ComplexType*>* val) {
		NDArray<double>* arr = (new Vector<double>(val->getSize()))->shapeLike<double>(val->getShape());

		for (int i = 0; i < val->getSize(); i++) {
			ComplexType* current = val->get(i);
			arr->set(i, current->real);
		}

		return arr;
	}

	inline NDArray<double>* imag(NDArray<ComplexType*>* val) {
		NDArray<double>* arr = (new Vector<double>(val->getSize()))->shapeLike<double>(val->getShape());

		for (int i = 0; i < val->getSize(); i++) {
			ComplexType* current = val->get(i);
			arr->set(i, current->real);
		}

		return arr;
	}

	inline NDArray<ComplexType*>* conj(NDArray<ComplexType*>* x) {
		NDArray<ComplexType*>* arr = x->emptyLike();

		for (int i = 0; i < x->getSize(); i++) {
			ComplexType* current = x->get(i);
			arr->set(i, new ComplexType(current->real, -current->imag));
		}

		return arr;
	}

	inline NDArray<ComplexType*>* conjugate(NDArray<ComplexType*>* x) {
		return conj(x);
	}

	inline NDArray<double*>* complexAbs(NDArray<ComplexType*>* x) {
		NDArray<double>* out = (new Vector<double>(x->getSize()))->shapeLike<double>(x->getShape());

		for (int i = 0; i < x->getSize(); i++) {
			ComplexType* current = x->get(i);
			double x2 = std::pow(current->real, 2);
			double y2 = std::pow(current->imag, 2);
			out->set(i, std::sqrt(x2 + y2));
		}
	}
	
	inline NDArray<ComplexType*>* complexSign(NDArray<ComplexType*>* x) {
		NDArray<ComplexType*>* out = x->emptyLike();

		for (int i = 0; i < x->getSize(); i++) {
			double real = x->get(i)->real;
			double newReal = (real / std::sqrt(std::pow(real, 2)));
			out->set(i, new ComplexType(real, 0));
		}

		return out;
	}
}