#pragma once

#include <cmath>
#include "../../Constants.cuh"
#include "../../Arrays/Arrays.cuh"

using Cudheart::NDArrays::Vector;
using Cudheart::NDArrays::Matrix;
using namespace Cudheart::Exceptions;
using Cudheart::VectorOps::emptyLike;
using Cudheart::MatrixOps::fromVector;
using namespace std;
using Cudheart::Constants::euler;
using Cudheart::Constants::pi;

namespace Cudheart::CPP::Math::Trigo {
	template <typename T>
	NDArray<T>* sin(NDArray<T>* rads) {
		NDArray<T>* out = rads->emptyLike();
		for (int i = 0; i < rads->getSize(); i++) {
			out->set(i, std::sin(rads->get(i)));
		}

		return out;
	}

	template <typename T>
	NDArray<T>* cos(NDArray<T>* rads) {
		NDArray<T>* out = rads->emptyLike();

		for (int i = 0; i < rads->getSize(); i++) {
			out->set(i, std::cos(rads->get(i)));
		}

		return out;
	}

	template <typename T>
	NDArray<T>* tan(NDArray<T>* rads) {
		NDArray<T>* out = rads->emptyLike();

		for (int i = 0; i < rads->getSize(); i++) {
			out->set(i, std::tan(rads->get(i)));
		}

		return out;
	}

	template <typename T>
	NDArray<T>* cot(NDArray<T>* rads) {
		NDArray<T>* out = rads->emptyLike();

		for (int i = 0; i < rads->getSize(); i++) {
			out->set(i, 1 / std::tan(rads->get(i)));
		}

		return out;
	}

	template <typename T>
	NDArray<T>* arcsin(NDArray<T>* x) {
		NDArray<T>* out = x->emptyLike();
		for (int i = 0; i < x->getSize(); i++) {
			out->set(i, asin(x->get(i)));
		}
		return out;
	}

	template <typename T>
	NDArray<T>* arccos(NDArray<T>* x) {
		NDArray<T>* out = x->emptyLike();
		for (int i = 0; i < x->getSize(); i++) {
			out->set(i, acos(x->get(i)));
		}
		return out;
	}

	template <typename T>
	NDArray<T>* arctan(NDArray<T>* x) {
		NDArray<T>* out = x->emptyLike();
		for (int i = 0; i < x->getSize(); i++) {
			out->set(i, atan(x->get(i)));
		}
		return out;
	}

	template <typename T>
	NDArray<T>* arccot(NDArray<T>* x) {
		NDArray<T>* out = x->emptyLike();
		for (int i = 0; i < x->getSize(); i++) {
			out->set(i, (pi / 2) - atan(x->get(i)));
		}
		return out;
	}

	template <typename T>
	NDArray<T>* hypot(NDArray<T>* a, NDArray<T>* b) {
		a->assertMatchShape(b);

		NDArray<T>* out = a->emptyLike();

		for (int i = 0; i < a->getSize(); i++) {
			T va = a->get(i);
			T vb = b->get(i);
			out->set(i, sqrt((va * va) + (vb * vb)));
		}

		return out;
	}

	template <typename T>
	NDArray<T>* deg2rad(NDArray<T>* degs) {
		NDArray<T>* out = degs->emptyLike();

		for (int i = 0; i < out->getSize(); i++) {
			out->set(i, degs->get(i) * (pi / 180));
		}
	}

	template <typename T>
	NDArray<T>* rad2deg(NDArray<T>* rads) {
		NDArray<T>* out = rads->emptyLike();

		for (int i = 0; i < out->getSize(); i++) {
			out->set(i, rads->get(i) / (pi / 180));
		}
	}

	template <typename T>
	NDArray<T>* sinc(NDArray<T>* rads) {
		NDArray<T>* out = rads->emptyLike();
		for (int i = 0; i < rads->getSize(); i++) {
			if (rads->get(i) == 0) {
				ZeroDivisionException("sinc");
			}
			out->set(i, std::sin(rads->get(i)) / rads->get(i));
		}

		return out;
	}

	template <typename T>
	NDArray<T>* sinh(NDArray<T>* x) {
		NDArray<T>* out = x->emptyLike();

		for (int i = 0; i < x->getSize(); i++) {
			out->set(i, std::sinh(x->get(i)));
		}

		return out;
	}

	template <typename T>
	NDArray<T>* cosh(NDArray<T>* x) {
		NDArray<T>* out = x->emptyLike();

		for (int i = 0; i < x->getSize(); i++) {
			out->set(i, std::cosh(x->get(i)));
		}

		return out;
	}

	template <typename T>
	NDArray<T>* tanh(NDArray<T>* x) {
		NDArray<T>* out = x->emptyLike();

		for (int i = 0; i < x->getSize(); i++) {
			out->set(i, std::tanh(x->get(i)));
		}

		return out;
	}

	template <typename T>
	NDArray<T>* arcsinh(NDArray<T>* x) {
		NDArray<T>* out = x->emptyLike();

		for (int i = 0; i < x->getSize(); i++) {
			out->set(i, std::asinh(x->get(i)));
		}

		return out;
	}

	template <typename T>
	NDArray<T>* arccosh(NDArray<T>* x) {
		NDArray<T>* out = x->emptyLike();

		for (int i = 0; i < x->getSize(); i++) {
			out->set(i, std::acosh(x->get(i)));
		}

		return out;
	}

	template <typename T>
	NDArray<T>* arctanh(NDArray<T>* x) {
		NDArray<T>* out = x->emptyLike();

		for (int i = 0; i < x->getSize(); i++) {
			out->set(i, std::atanh(x->get(i)));
		}

		return out;
	}
}