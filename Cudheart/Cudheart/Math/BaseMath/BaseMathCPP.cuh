#pragma once

#include "../../Arrays/Arrays.cuh"
#include <cmath>
#include <numeric>

using Cudheart::NDArrays::Vector;
using Cudheart::NDArrays::Matrix;
using namespace Cudheart::Exceptions;
using Cudheart::VectorOps::emptyLike;
using Cudheart::MatrixOps::fromVector;

namespace Cudheart::CPP::Math::BaseMath {
	template <typename T>
	NDArray<T>* cubeRoot(NDArray<T>* arr) {
		NDArray<T>* output = arr->emptyLike();

		for (int i = 0; i < arr->size(); i++) {
			output->set(i, cbrt(arr->get(i)));
		}

		return output;
	}

	template <typename T>
	NDArray<T>* cube(NDArray<T>* base) {
		return power(base, 3);
	}

	template <typename T>
	NDArray<T>* square(NDArray<T>* base) {
		return power(base, 2);
	}

	template <typename T>
	NDArray<T>* squareRoot(NDArray<T>* arr) {
		NDArray<T>* output = arr->emptyLike();

		for (int i = 0; i < arr->size(); i++) {
			output->set(i, sqrt(arr->get(i)));
		}

		return output;
	}

	template <typename T>
	NDArray<T>* power(NDArray<T>* base, T po) {
		NDArray<T>* output = base->emptyLike();

		for (int i = 0; i < base->size(); i++) {
			output->set(i, pow(base->get(i), po));
		}

		return output;
	}

	template <typename T>
	NDArray<T>* power(T base, NDArray<T>* po) {
		NDArray<T>* output = po->emptyLike();

		for (int i = 0; i < po->size(); i++) {
			output->set(i, pow(base, po->get(i)));
		}

		return output;
	}

	template <typename T>
	NDArray<T>* power(NDArray<T>* base, NDArray<T>* po) {
		base->assertMatchShape(po->getShape());

		NDArray<T>* output = base->emptyLike();

		for (int i = 0; i < base->size(); i++) {
			output->set(i, pow(base->get(i), po->get(i)));
		}

		return output;
	}

	template <typename T>
	Matrix<T>* power(Matrix<T>* base, Vector<T>* po) {
		base->assertMatchShape(po->getShape(), 1);

		Matrix<T>* out = emptyLike<T>(base);

		for (int i = 0; i < base->getHeight(); i++) {
			T p = po->get(i);
			for (int j = 0; j < base->getWidth(); j++) {
				out->set(i, j, pow(p, base->get(i, j)));
			}
		}

		return out;
	}

	template <typename T>
	NDArray<T>* around(NDArray<T>* arr, int decimals) {
		NDArray<T>* out = arr->emptyLike();
		int v = std::pow(10, decimals);

		for (int i = 0; i < out->size(); i++) {
			T value = arr->get(i) * v;
			value = std::round(value);
			out->set(i, value / v);
		}

		return out;
	}

	template <typename T>
	NDArray<T>* around(NDArray<T>* arr) {
		return around(arr, 0);
	}

	template <typename T>
	NDArray<T>* rint(NDArray<T>* arr) {
		NDArray<T>* out = arr->emptyLike();

		for (int i = 0; i < out->size(); i++) {
			out->set(i, std::rint(arr->get(i)));
		}

		return out;
	}

	template <typename T>
	NDArray<T>* fix(NDArray<T>* x) {
		NDArray<T>* out = x->emptyLike();

		for (int i = 0; i < x->size(); i++) {
			T v = x->get(i);
			if (v > 0) {
				out->set(i, std::floor(v));
			}
			else {
				out->set(i, std::ceil(v));
			}
		}

		return out;
	}

	template <typename T>
	NDArray<T>* floor(NDArray<T>* x) {
		NDArray<T>* out = x->emptyLike();

		for (int i = 0; i < x->size(); i++) {
			out->set(i, std::floor(x->get(i)));
		}

		return out;
	}

	template <typename T>
	NDArray<T>* trunc(NDArray<T>* x) {
		NDArray<T>* out = x->emptyLike();

		for (int i = 0; i < x->size(); i++) {
			out->set(i, std::trunc(x->get(i)));
		}

		return out;
	}

	template <typename T>
	NDArray<T>* ceil(NDArray<T>* x) {
		NDArray<T>* out = x->emptyLike();

		for (int i = 0; i < x->size(); i++) {
			out->set(i, std::ceil(x->get(i)));
		}

		return out;
	}
	template <typename T>
	T prod(NDArray<T>* x) {
		T out = (T)1;

		for (int i = 0; i < x->size(); i++) {
			out *= x->get(i);
		}

		return out;
	}

	template <typename T>
	T sum(NDArray<T>* x) {
		T out = (T)0;

		for (int i = 0; i < x->size(); i++) {
			out += x->get(i);
		}

		return out;
	}

	template <typename T>
	NDArray<T>* cumProd(NDArray<T>* x) {
		NDArray<T>* out = x->emptyLike();

		T prod = (T)1;

		for (int i = 0; i < x->size(); i++) {
			prod *= x->get(i);
			out->set(i, prod);
		}

		return out;
	}

	template <typename T>
	NDArray<T>* cumSum(NDArray<T>* x) {
		NDArray<T>* out = x->emptyLike();

		T prod = (T)0;

		for (int i = 0; i < x->size(); i++) {
			prod += x->get(i);
			out->set(i, prod);
		}

		return out;
	}

	template <typename T>
	NDArray<bool>* signBit(NDArray<T>* x) {
		NDArray<bool>* out = x->emptyLike<bool>();

		for (int i = 0; i < x->size(); i++) {
			T v = x->get(i);
			out->set(i, v < 0);
		}

		return out;
	}

	template <typename T>
	NDArray<T>* copySign(NDArray<T>* a, NDArray<T>* b) {
		a->assertMatchShape(b->getShape());

		NDArray<T>* out = a->emptyLike();

		for (int i = 0; i < a->size(); i++) {
			T va = a->get(i);
			T vb = b->get(i);

			// what tf is wrong with me
			if ((vb < 0) != (va < 0)) {
				va = -va;
			}
			out->set(i, va);
		}

		return out;
	}

	template <typename T>
	NDArray<T>* abs(NDArray<T>* x) {
		NDArray<T>* out = x->emptyLike();

		for (int i = 0; i < x->size(); i++) {
			out->set(i, std::abs(x->get(i)));
		}

		return out;
	}

	template <typename T>
	NDArray<T>* lcm(NDArray<T>* a, NDArray<T>* b) {
		a->assertMatchShape(b->getShape());
		NDArray<T>* out = a->emptyLike();

		for (int i = 0; i < a->size(); i++) {
			out->set(i, std::lcm(a->get(i), b->get(i)));
		}

		return out;
	}

	template <typename T>
	NDArray<T>* gcd(NDArray<T>* a, NDArray<T>* b) {
		a->assertMatchShape(b->getShape());
		NDArray<T>* out = a->emptyLike();

		for (int i = 0; i < a->size(); i++) {
			out->set(i, std::gcd(a->get(i), b->get(i)));
		}

		return out;
	}

	template <typename T>
	NDArray<T>* add(NDArray<T>* a, NDArray<T>* b) {
		a->assertMatchShape(b->getShape());
		NDArray<T>* out = a->emptyLike();

		for (int i = 0; i < a->size(); i++) {
			out->set(i, a->get(i) + b->get(i));
		}

		return out;
	}

	template <typename T>
	NDArray<T>* subtract(NDArray<T>* a, NDArray<T>* b) {
		a->assertMatchShape(b->getShape());
		NDArray<T>* out = a->emptyLike();

		for (int i = 0; i < a->size(); i++) {
			out->set(i, a->get(i) - b->get(i));
		}

		return out;
	}

	template <typename T>
	NDArray<T>* multiply(NDArray<T>* a, NDArray<T>* b) {
		a->assertMatchShape(b->getShape());
		NDArray<T>* out = a->emptyLike();

		for (int i = 0; i < a->size(); i++) {
			out->set(i, a->get(i) * b->get(i));
		}

		return out;
	}

	template <typename T>
	NDArray<T>* divide(NDArray<T>* a, NDArray<T>* b) {
		a->assertMatchShape(b->getShape());
		NDArray<T>* out = a->emptyLike();

		for (int i = 0; i < a->size(); i++) {
			out->set(i, a->get(i) / b->get(i));
		}

		return out;
	}

	template <typename T>
	NDArray<int>* floorDivide(NDArray<T>* a, NDArray<T>* b) {
		a->assertMatchShape(b->getShape());
		NDArray<int>* out = a->emptyLike<int>();

		for (int i = 0; i < a->size(); i++) {
			out->set(i, std::floor(a->get(i) / b->get(i)));
		}

		return out;
	}

	template <typename T>
	NDArray<T>* mod(NDArray<T>* a, NDArray<T>* b) {
		a->assertMatchShape(b->getShape());
		NDArray<T>* out = a->emptyLike();

		for (int i = 0; i < a->size(); i++) {
			out->set(i, a->get(i) % b->get(i));
		}

		return out;
	}

	template <typename T>
	NDArray<T>** divMod(NDArray<T>* a, NDArray<T>* b) {
		a->assertMatchShape(b->getShape());
		NDArray<T>* div = a->emptyLike();
		NDArray<T>* mod = a->emptyLike();

		for (int i = 0; i < a->size(); i++) {
			int d = a->get(i) / b->get(i);
			div->set(i, d);
			mod->set(i, a->get(i) - (d * b->get(i)));
		}

		NDArray<T>** out = (NDArray<T>**)malloc(sizeof(NDArray<T>) * 2);
		out[0] = div;
		out[1] = mod;

		return out;
	}

	template <typename T>
	NDArray<T>* reciprocal(NDArray<T>* x) {
		NDArray<T>* out = x->emptyLike();

		for (int i = 0; i < x->size(); i++) {
			out->set(i, (T)1 / x->get(i));
		}

		return out;
	}

	template <typename T>
	NDArray<T>* positive(NDArray<T>* x) {
		NDArray<T>* out = x->emptyLike();

		for (int i = 0; i < x->size(); i++) {
			out->set(i, +x->get(i));
		}

		return out;
	}

	template <typename T>
	NDArray<T>* negative(NDArray<T>* x) {
		NDArray<T>* out = x->emptyLike();

		for (int i = 0; i < x->size(); i++) {
			out->set(i, -x->get(i));
		}

		return out;
	}

	template <typename T>
	NDArray<T>* sign(NDArray<T>* x) {
		NDArray<T>* out = x->emptyLike();

		for (int i = 0; i < x->size(); i++) {
			if (std::signbit(x->get(i))) {
				out->set(i, -1);
			}
			else {
				out->set(i, 1);
			}
		}

		return out;
	}

	template <typename T>
	NDArray<T>* heaviside(NDArray<T>* a, NDArray<T>* b) {
		a->assertMatchShape(b->getShape());
		NDArray<T>* out = a->emptyLike();

		for (int i = 0; i < a->size(); i++) {
			T x1 = a->get(i);
			if (x1 < 0) {
				out->set(i, 0);
			}
			else if (x1 == 0) {
				out->set(i, b->get(i));
			}
			else if (x1 > 0) {
				out->set(i, 1);
			}
		}

		return out;
	}
}