#pragma once

#include <time.h>
#include <stdlib.h>

#include "../../Arrays/Arrays.cuh"

using Cudheart::NDArrays::Vector;
using Cudheart::NDArrays::Matrix;

namespace Cudheart::CPP::Random {
	inline void seed(unsigned seed) {
		srand(seed);
	}

	inline void seed() {
		seed(time(NULL));
	}

	inline Vector<int>* integers(int low, int high, int size, bool endpoint) {
		seed();
		Vector<int>* out = new Vector<int>(size);

		for (int i = 0; i < size; i++) {
			if (endpoint) {
				out->set(i, rand() % (high - low + 1) + low);
			} else {
				out->set(i, rand() % (high - low) + low);
			}
		}

		return out;
	}
	
	inline Vector<int>* integers(int low, int high, int size) {
		return integers(low, high, size, false);
	}

	inline Vector<int>* integers(int high, int size, bool endpoint) {
		return integers(0, high, size, endpoint);
	}

	inline Vector<int>* integers(int high, int size) {
		return integers(0, high, size, false);
	}

	inline Matrix<int>* integers2d(int low, int high, bool endpoint, int width, int height) {
		seed();
		int size = width * height;
		Matrix<int>* out = new Matrix<int>(width, height);

		for (int i = 0; i < size; i++) {
			if (endpoint) {
				out->set(i, rand() % (high - low + 1) + low);
			}
			else {
				out->set(i, rand() % (high - low) + low);
			}
		}

		return out;
	}

	inline Matrix<int>* integers2d(int low, int high, int width, int height) {
		return integers2d(low, high, false, width, height);
	}

	inline Matrix<int>* integers2d(int high, bool endpoint, int width, int height) {
		return integers2d(0, high, endpoint, width, height);
	}

	inline Matrix<int>* integers2d(int high, int width, int height) {
		return integers2d(0, high, false, width, height);
	}

	template <typename T>
	inline Vector<T>* random(int size) {
		seed();
		Vector<T>* out = new Vector<T>(size);

		for (int i = 0; i < size; i++) {
			out->set(i, (T)rand() / (T)RAND_MAX);
		}

		return out;
	}

	template <typename T>
	inline Matrix<T>* random(int height, int width) {
		seed();
		Matrix<T>* out = new Matrix<T>(height, width);

		for (int i = 0; i < out->getSize(); i++) {
			out->set(i, (T)rand() / (T)RAND_MAX);
		}

		return out;
	}

	inline double random() {
		seed();
		return rand() / RAND_MAX;
	}
	
	template <typename T>
	inline Vector<T>* choice(NDArray<T>* a, int size, bool replace, Vector<double>* p) {
		seed();
		Vector<T>* vec = new Vector<T>(size);
		p->assertMatchShape(a->getShape());

		// assert that size < a->getSize() if replace is false

		for (int i = 0; i < size; i++) {
			double prob = random();
			int j = 0;
			while (prob > 0) {
				prob -= p->get(j++);
			}
			if (!replace) {
				p->set(j, 0);
			}
			vec->set(i, a->get(j));
		}

		return vec;
	}

	template <typename T>
	inline Vector<T>* choice(NDArray<T>* a, int size) {
		Vector<double>* p = new Vector(a->getSize());

		for (int i = 0; i < p->getSize(); i++) {
			p->set(i, 1 / p->getSize());
		}

		return choice<T>(a, size, p);
	}
}