#pragma once


using Cudheart::NDArrays::Vector;
using Cudheart::NDArrays::Matrix;
using namespace Cudheart::Exceptions;
using Cudheart::VectorOps::empty;
using Cudheart::VectorOps::emptyLike;
using Cudheart::MatrixOps::fromVector;

namespace Cudheart::CPP::Math::Trigo {
	template <typename T>
	Vector<T>* sin(Vector<T>* rads) {
		Vector<T>* out = emptyLike<T>(rads);
		
		for (int i = 0; i < rads->size(); i++) {
			out->at(i) = sin(rads->at(i));
		}

		return out;
	}

	template <typename T>
	Matrix<T>* sin(Matrix<T>* rads) {
		Matrix<T>* out = emptyLike<T>(rads);

		for (int i = 0; i < rads->size(); i++) {
			out->at(i) = sin(rads->at(i));
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
}