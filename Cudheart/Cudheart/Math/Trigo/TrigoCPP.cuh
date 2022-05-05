#pragma once

#include <cmath>

using Cudheart::NDArrays::Vector;
using Cudheart::NDArrays::Matrix;
using namespace Cudheart::Exceptions;
using Cudheart::VectorOps::empty;
using Cudheart::VectorOps::emptyLike;
using Cudheart::MatrixOps::fromVector;
using namespace std;

namespace Cudheart::CPP::Math::Trigo {
	template <typename T>
	Vector<T>* sin(Vector<T>* rads) {
		Vector<T>* out = emptyLike<T>(rads);
		for (int i = 0; i < rads->size(); i++) {
			out->set(i, std::sin(rads->get(i)));
		}

		return out;
	}

	template <typename T>
	Matrix<T>* sin(Matrix<T>* rads) {
		Matrix<T>* out = emptyLike<T>(rads);

		for (int i = 0; i < rads->size(); i++) {
			out->set(i, std::sin(rads->get(i)));
		}

		return out;
	}

	template <typename T>
	Vector<T>* cos(Vector<T>* rads) {
		Vector<T>* out = emptyLike<T>(rads);

		for (int i = 0; i < rads->size(); i++) {
			out->set(i, std::cos(rads->get(i)));
		}

		return out;
	}

	template <typename T>
	Matrix<T>* cos(Matrix<T>* rads) {
		Matrix<T>* out = emptyLike<T>(rads);

		for (int i = 0; i < rads->size(); i++) {
			out->set(i, std::cos(rads->get(i)));
		}

		return out;
	}

	template <typename T>
	Vector<T>* tan(Vector<T>* rads) {
		Vector<T>* out = emptyLike<T>(rads);

		for (int i = 0; i < rads->size(); i++) {
			out->set(i, std::tan(rads->get(i)));
		}

		return out;
	}

	template <typename T>
	Matrix<T>* tan(Matrix<T>* rads) {
		Matrix<T>* out = emptyLike<T>(rads);

		for (int i = 0; i < rads->size(); i++) {
			out->set(i, std::tan(rads->get(i)));
		}

		return out;
	}

	template <typename T>
	Vector<T>* cot(Vector<T>* rads) {
		Vector<T>* out = emptyLike<T>(rads);

		for (int i = 0; i < rads->size(); i++) {
			out->set(i, 1 / std::tan(rads->get(i)));
		}

		return out;
	}

	template <typename T>
	Matrix<T>* cot(Matrix<T>* rads) {
		Matrix<T>* out = emptyLike<T>(rads);

		for (int i = 0; i < rads->size(); i++) {
			out->set(i, 1 / std::tan(rads->get(i)));
		}

		return out;
	}
	
	template <typename T>
	Vector<T>* arccos(Vector<T>* vec) {
		Vector<T>* out = emptyLike(vec);
		for (int i = 0; i < vec->getSize(); i++) {
			out->set(i, acos(vec->get(i)));
		}
		return out;
	}

	template <typename T>
	Matrix<T>* arccos(Matrix<T>* mat) {
		Vector<T>* flat = mat->flatten();

		Vector<T>* out = arccos(flat);

		delete flat;

		return fromVector(out, mat->getWidth(), mat->getHeight(), true);
	}

	template <typename T>
	Vector<T>* arcsin(Vector<T>* vec) {
		Vector<T>* out = emptyLike(vec);
		for (int i = 0; i < vec->getSize(); i++) {
			out->set(i, asin(vec->get(i)));
		}
		return out;
	}

	template <typename T>
	Matrix<T>* arcsin(Matrix<T>* mat) {
		Vector<T>* flat = mat->flatten();

		Vector<T>* out = arccos(flat);

		delete flat;

		return fromVector(out, mat->getWidth(), mat->getHeight(), true);
	}

	template <typename T>
	Vector<T>* arctan(Vector<T>* vec) {
		Vector<T>* out = emptyLike(vec);
		for (int i = 0; i < vec->getSize(); i++) {
			out->set(i, atan(vec->get(i)));
		}
		return out;
	}

	template <typename T>
	Matrix<T>* arctan(Matrix<T>* mat) {
		Vector<T>* flat = mat->flatten();

		Vector<T>* out = arctan(flat);

		delete flat;

		return fromVector(out, mat->getWidth(), mat->getHeight(), true);
	}

	template <typename T>
	Vector<T>* arccot(Vector<T>* vec) {
		Vector<T>* out = emptyLike(vec);
		long double pi = 3.1415926535897932384626433;
		for (int i = 0; i < vec->getSize(); i++) {
			out->set(i, (pi / 2) - atan(vec->get(i)));
		}
		return out;
	}

	template <typename T>
	Matrix<T>* arccot(Matrix<T>* mat) {
		Vector<T>* flat = mat->flatten();

		Vector<T>* out = arccot(flat);

		delete flat;

		return fromVector(out, mat->getWidth(), mat->getHeight(), true);
	}

	template <typename T>
	Vector<T>* hypot(Vector<T>* a, Vector<T>* b) {
		a->assertMatchSize(b);

		Vector<T>* out = emptyLike(a);

		for (int i = 0; i < a->getSize(); i++) {
			T va = a->get(i);
			T vb = b->get(i);
			out->set(i, sqrt((va * va) + (vb * vb)));
		}

		return out;
	}

	template <typename T>
	Matrix<T>* hypot(Matrix<T>* a, Matrix<T>* b) {
		a->assertMatchSize(b);

		Matrix<T>* out = emptyLike(a);

		for (int i = 0; i < a->getSize(); i++) {
			T va = a->get(i);
			T vb = b->get(i);
			out->set(i, sqrt((va * va) + (vb * vb)));
		}

		return out;
	}
}