#pragma once

#include <time.h>
#include <stdlib.h>
#include <random>

#include "../Arrays/Arrays.cuh"

using Cudheart::NDArrays::Vector;
using Cudheart::NDArrays::Matrix;
using random_bytes_engine = std::independent_bits_engine<std::default_random_engine, CHAR_BIT, unsigned char>;
using namespace Cudheart::Constants;

namespace Cudheart::CPP::Random {
	inline void seed(unsigned seed) {
		Constants::setSeed(seed);
	}

	inline Vector<int>* integers(int low, int high, int size, bool endpoint) {
		srand(Constants::getSeed());
		
		Vector<int>* out = new Vector<int>(size);

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
		srand(Constants::getSeed());
		
		Vector<T>* out = new Vector<T>(size);

		for (int i = 0; i < size; i++) {
			out->set(i, (T)rand() / (T)RAND_MAX);
		}

		return out;
	}

	template <typename T>
	inline Matrix<T>* random(int height, int width) {
		srand(Constants::getSeed());
		
		Matrix<T>* out = new Matrix<T>(height, width);

		for (int i = 0; i < out->getSize(); i++) {
			out->set(i, (T)rand() / (T)RAND_MAX);
		}

		return out;
	}

	inline double random() {
		srand(Constants::getSeed());
		return rand() / RAND_MAX;
	}

	template <typename T>
	inline Vector<T>* choice(NDArray<T>* a, int size, bool replace, Vector<double>* p) {
		srand(Constants::getSeed());
		Vector<T>* vec = new Vector<T>(size);
		p->assertMatchShape(a->getShape());

		// assert that size < a->getSize() if replace is false

		for (int i = 0; i < size; i++) {
			double prob = rand() / RAND_MAX;
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
		srand(Constants::getSeed());
		Vector<double>* p = new Vector(a->getSize());

		for (int i = 0; i < p->getSize(); i++) {
			p->set(i, 1 / p->getSize());
		}

		return choice<T>(a, size, p);
	}

	template <typename T>
	inline Vector<T>* bytes(int length) {
		srand(Constants::getSeed());
		random_bytes_engine engine;
		engine.seed(Constants::getSeed());
		std::vector<unsigned char> data(length);
		std::generate(begin(data), end(data), std::ref(engine));

		Vector<T>* vec = new Vector<T>(length);

		for (int i = 0; i < length; i++) {
			vec->set(i, data.at(i));
		}

		return vec;
	}

	template <typename T>
	inline Matrix<T>* bytes(int height, int width) {
		srand(Constants::getSeed());
		random_bytes_engine engine;
		engine.seed(Constants::getSeed());
		std::vector<unsigned char> data(height * width);
		std::generate(begin(data), end(data), std::ref(engine));

		Matrix<T>* mat = new Matrix<T>(height, width);

		for (int i = 0; i < mat->getSize(); i++) {
			mat->set(i, data.at(i));
		}

		return mat;
	}

	template <typename T>
	inline void shuffle(NDArray<T>* a) {
		srand(Constants::getSeed());
		NDArray<T>* copy = a->copy();
		int size = a->getSize();
		int* indices = new int[size];
		for (int i = 0; i < size; i++) {
			indices[i] = i;
		}
		for (int i = 0; i < size; i++) {
			int j = rand() % size;
			int temp = indices[i];
			indices[i] = indices[j];
			indices[j] = temp;
		}
		for (int i = 0; i < size; i++) {
			a->set(i, copy->get(indices[i]));
		}
		delete[] indices;
	}

	template <typename T>
	inline NDArray<T>* permutation(NDArray<T>* a) {
		NDArray<T>* out = a->copy();
		shuffle<T>(out);
		return out;
	}

	template <typename T>
	inline Vector<T>* permutation(int x) {
		return permutation<T>(Cudheart::VectorOps::arange<T>(x));
	}

	template <typename T>
	inline NDArray<T>* beta(NDArray<T>* a, NDArray<T>* b) {
		a->assertMatchShape(b);
		NDArray<T>* out = a->copy();

		for (int i = 0; i < a->getSize(); i++) {
			out->set(i, std::beta(a->get(i), b->get(i)));
		}

		return out;
	}

	template <typename T>
	inline NDArray<T>* binomal(NDArray<T>* a, NDArray<T>* b) {
		srand(getSeed());
		a->assertMatchShape(b);
		NDArray<T>* out = a->copy();
		default_random_engine generator(getSeed());
		std::binomial_distribution<T> dist;

		for (int i = 0; i < a->getSize(); i++) {
			out->set(i, dist(generator));
		}

		return out;
	}

	template <typename T>
	inline NDArray<T>* chisquare(NDArray<T>* x) {
		srand(getSeed());
		NDArray<T>* out = x->copy();
		default_random_engine generator(getSeed());
		std::chi_squared_distribution<T> dist;

		for (int i = 0; i < x->getSize(); i++) {
			out->set(i, dist(generator));
		}

		return out;
	}

	template <typename T>
	inline NDArray<T>* dirichlet(NDArray<T>* alpha) {
		srand(getSeed());
		NDArray<T>* out = alpha->emptyLike();
		default_random_engine generator(getSeed());
		T sum = 0;

		for (int i = 0; i < alpha->getSize(); i++) {
			std::gamma_distribution<> temp(alpha->get(i), 1);
			T x = temp(generator);
			out->set(i, x);
			sum += x;
		}

		for (int i = 0; i < out->getSize(); i++) {
			out->set(i, out->get(i) / sum);
		}

		return out;
	}

	template <typename T>
	inline NDArray<T>* exponential(NDArray<T>* x) {
		srand(getSeed());
		NDArray<T>* out = x->emptyLike();
		default_random_engine generator(getSeed());
		std::exponential_distribution<T> dist;

		for (int i = 0; i < x->getSize(); i++) {
			out->set(i, dist(generator));
		}

		return out;
	}

	template <typename T>
	inline NDArray<T>* f(NDArray<T>* dfnum, NDArray<T>* dfden) {
		srand(getSeed());
		dfnum->assertMatchShape(dfden);
		NDArray<T>* out = dfnum->emptyLike();
		default_random_engine generator(getSeed());

		for (int i = 0; i < dfnum->getSize(); i++) {
			std::fisher_f_distribution dist(dfnum->get(i), dfden->get(i));
			out->set(i, dist(generator));
		}

		return out;
	}
}