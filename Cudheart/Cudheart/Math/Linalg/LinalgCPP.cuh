#pragma once

#include "../../Arrays/Arrays.cuh"

using Cudheart::NDArrays::Vector;
using Cudheart::NDArrays::Matrix;
using namespace Cudheart::Exceptions;
using Cudheart::VectorOps::empty;
using Cudheart::VectorOps::emptyLike;
using Cudheart::MatrixOps::empty;
using Cudheart::MatrixOps::emptyLike;


namespace Cudheart::CPP::Math::Linalg {
	template <typename T>
	T dot(Vector<T>* a, Vector<T>* b) {
		a->assertMatchSize(b);

		T result = 0;

		for (int i = 0; i < a->getSize(); i++) {
			result += a->get(i) * b->get(i);
		}

		return result;
	}

	template <typename T>
	Vector<T>* dot(Vector<T>* a, Matrix<T>* b) {
		if (b->getHeight() != a->getSize()) {
			ShapeMismatchException(b->getHeight(), a->getSize()).raise();
		}

		Matrix<T>* t = emptyLike<T>(b);

		for (int i = 0; i < b->getHeight(); i++) {
			for (int j = 0; j < b->getWidth(); j++) {
				t->set(i, j, b->get(i, j) * a->get(i));
			}
		}

		Vector<T>* out = emptyLike<T>(a);

		for (int i = 0; i < b->getWidth(); i++) {
			int sum = 0;
			for (int j = 0; j < b->getHeight(); j++) {
				sum += t->get(j, i);
			}
			out->set(i, sum);
		}

		delete t;

		return out;
	}

	template <typename T>
	Vector<T>* dot(Matrix<T>* a, Vector<T>* b) {
		if (a->getWidth() != b->getSize()) {
			ShapeMismatchException(a->getWidth(), b->getSize()).raise();
		}

		Matrix<T>* t = emptyLike<T>(a);

		for (int i = 0; i < a->getHeight(); i++) {
			for (int j = 0; j < a->getWidth(); j++) {
				t->set(i, j, a->get(i, j) * b->get(j));
			}
		}

		Vector<T>* out = empty<T>(a->getHeight());

		for (int i = 0; i < a->getHeight(); i++) {
			int sum = 0;
			for (int j = 0; j < a->getWidth(); j++) {
				sum += t->get(i, j);
			}
			out->set(i, sum);
		}

		return out;
	}
}