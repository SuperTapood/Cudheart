#pragma once

#include <time.h>
#include <stdlib.h>
#include <random>

#include "../Arrays/Arrays.cuh"
#include "../Constants.cuh"

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

		for (int i = 0; i < out->size(); i++) {
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
		Vector<double>* p = new Vector(a->size());

		for (int i = 0; i < p->size(); i++) {
			p->set(i, 1 / p->size());
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

		for (int i = 0; i < mat->size(); i++) {
			mat->set(i, data.at(i));
		}

		return mat;
	}

	template <typename T>
	inline void shuffle(NDArray<T>* a) {
		srand(Constants::getSeed());
		NDArray<T>* copy = a->copy();
		int size = a->size();
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

		for (int i = 0; i < a->size(); i++) {
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

		for (int i = 0; i < a->size(); i++) {
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

		for (int i = 0; i < x->size(); i++) {
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

		for (int i = 0; i < alpha->size(); i++) {
			std::gamma_distribution<> temp(alpha->get(i), 1);
			T x = temp(generator);
			out->set(i, x);
			sum += x;
		}

		for (int i = 0; i < out->size(); i++) {
			out->set(i, out->get(i) / sum);
		}

		return out;
	}

	template <typename T>
	inline NDArray<T>* exponential(NDArray<T>* x) {
		srand(getSeed());
		NDArray<T>* out = x->emptyLike();
		default_random_engine generator(getSeed());

		for (int i = 0; i < x->size(); i++) {
			std::exponential_distribution<T> dist(x->get(i));
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

		for (int i = 0; i < dfnum->size(); i++) {
			std::fisher_f_distribution dist(dfnum->get(i), dfden->get(i));
			out->set(i, dist(generator));
		}

		return out;
	}

	template <typename T>
	inline NDArray<T>* gamma(NDArray<T>* shape, NDArray<T>* scale) {
		srand(getSeed());
		shape->assertMatchShape(scale);
		NDArray<T>* out = shape->emptyLike();
		default_random_engine generator(getSeed());

		for (int i = 0; i < shape->size(); i++) {
			std::gamma_distribution<> dist(shape->get(i), scale->get(i));
			out->set(i, dist(generator));
		}

		return out;
	}

	template <typename T>
	inline Vector<T>* geometric(T p, int size) {
		srand(getSeed());
		Vector<T>* out = new Vector<T>(size);
		default_random_engine generator(getSeed());
		std::geometric_distribution<> dist(p);

		for (int i = 0; i < size; i++) {
			out->set(i, dist(generator));
		}
	}

	template <typename T>
	inline Vector<T>* geometric(Vector<T>* p, int size) {
		srand(getSeed());
		// assert size <= p.size()
		Vector<T>* out = new Vector<T>(size);
		default_random_engine generator(getSeed());

		for (int i = 0; i < size; i++) {
			std::geometric_distribution<> dist(p->get(i));
			out->set(i, dist(generator));
		}
	}

	template <typename T>
	inline Vector<T>* geometric(Vector<T>* p) {
		return geometric<T>(p, p->size());
	}

	template <typename T>
	inline Matrix<T>* geometric(Matrix<T>* p, int height, int width) {
		srand(getSeed());
		// assert size <= p.size()
		Matrix<T>* out = new Matrix<T>(height, width);
		default_random_engine generator(getSeed());

		for (int i = 0; i < out->size(); i++) {
			std::geometric_distribution<> dist(p->get(i));
			out->set(i, dist(generator));
		}
	}

	template <typename T>
	inline Matrix<T>* geometric(Matrix<T>* p) {
		return geometric<T>(p, p->getHeight(), p->getWidth());
	}

	template <typename T>
	inline NDArray<T>* gumbel(NDArray<T>* loc, NDArray<T>* scale) {
		loc->assertMatchShape(scale);
		srand(getSeed());
		default_random_engine generator(getSeed());
		NDArray<T>* out = loc->emptyLike();

		for (int i = 0; i < loc->size(); i++) {
			std::extreme_value_distribution<> dist(loc->get(i), scale->get(i));
			out->set(i, dist(generator));
		}

		return out;
	}

	// move somewhere else
	// using code from https://www.youtube.com/watch?v=o-ZtGGXLogE
	template <typename T>
	inline T binomal(T n, T k) {
		T ans = (T)1;

		if (k > n - k) {
			k = n - k;
		}

		for (int i = 0; i < k; i++) {
			ans *= (n - i);
			ans /= (i + 1);
		}

		return ans;
	}

	template <typename T>
	inline NDArray<T>* hypergeometric(NDArray<T>* ngood, NDArray<T>* nbad, NDArray<T>* nsample) {
		srand(getSeed());
		ngood->assertMatchShape(nbad);
		ngood->assertMatchShape(nsample);
		NDArray<T>* out = ngood->emptyLike();

		for (int i = 0; i < out->size(); i++) {
			T N = ngood->get(i) + nbad->get(i);
			T k = ngood->get(i);
			T n = nsample->get(i);
			T K = (T)rand() / (T)RAND_MAX;

			T above = binomal<T>(K, k) * binomal<T>(N - k, n - k);
			T below = binomal<T>(N, n);

			out->set(i, above / below);
		}

		return out;
	}

	template <typename T>
	inline NDArray<T>* laplace(NDArray<T>* loc, NDArray<T>* scale) {
		loc->assertMatchShape(scale);
		srand(getSeed());
		NDArray<T>* out = loc->emptyLike();

		for (int i = 0; i < loc->size(); i++) {
			T mew = loc->get(i);
			T lambda = scale->get(i);
			T x = (T)rand() / (T)RAND_MAX;
			T left = (T)1 / (2 * lambda);
			T right = std::exp(-((std::abs(x - mew) / lambda)));
			out->set(i, left * right);
		}

		return out;
	}

	template <typename T>
	inline NDArray<T>* logistic(NDArray<T>* loc, NDArray<T>* scale) {
		loc->assertMatchShape(scale);
		srand(getSeed());
		NDArray<T>* out = loc->emptyLike();

		for (int i = 0; i < loc->size(); i++) {
			T mew = loc->get(i);
			T s = scale->get(i);
			T x = (T)rand() / (T)RAND_MAX;
			T high = std::exp(-((x - mew) / s));
			T low = s * std::pow(1 + high, 2);
			out->set(i, high / low);
		}

		return out;
	}

	template <typename T>
	inline NDArray<T>* lognormal(NDArray<T>* mean, NDArray<T>* sigma) {
		mean->assertMatchShape(sigma);
		srand(getSeed());
		NDArray<T>* out = mean->emptyLike();

		for (int i = 0; i < mean->size(); i++) {
			T mew = mean->get(i);
			T o = sigma->get(i);
			T x = (T)rand() / (T)RAND_MAX;
			T left = 1 / (o * x * std::sqrt(2 * pi));
			T right = std::exp(-(std::pow(std::log(x) - mew, 2) / 2 * o * o));
			out->set(i, left * right);
		}
	}

	template <typename T>
	inline NDArray<T>* logseries(NDArray<T>* p) {
		srand(getSeed());
		NDArray<T>* out = p->emptyLike();

		for (int i = 0; i < p->size(); i++) {
			T k = (T)rand() / (T)RAND_MAX;
			T prob = p->get(i);
			T high = -std::pow(prob, k);
			T low = k * std::log(1 - prob);
			out->set(i, high / low);
		}

		return out;
	}

	template <typename T>
	inline NDArray<T>* multinominal(int n, NDArray<T>* pvals) {
		// assert that the sum of pvals is 1, and all values are between 0 and 1
		srand(getSeed());
		NDArray<T>* out = pvals->emptyLike();

		for (int i = 0; i < out->size(); i++) {
			out->set(i, 0);
		}

		for (int i = 0; i < n; i++) {
			T k = (T)rand() / (T)RAND_MAX;
			T prob = 0;
			for (int j = 0; j < pvals->size(); j++) {
				prob += pvals->get(j);
				if (k <= prob) {
					out->set(j, out->get(j) + 1);
					break;
				}
			}
		}

		return out;
	}

	template <typename T>
	inline NDArray<T>* negativeBinomal(NDArray<T>* n, NDArray<T>* p) {
		n->assertMatchShape(p);
		srand(getSeed());
		NDArray<T>* out = n->emptyLike();
		std::default_random_engine rng(getSeed());

		for (int i = 0; i < n->size(); i++) {
			T nv = n->get(i);
			T pv = p->get(i);
			T N = rand();
			T high = std::gamma_distribution<>()(nv + N);
			T low = std::tgamma(N + 1) * std::gamma_distribution<>()(nv);
			T right = std::pow(pv, n) * std::pow(1 - pv, N);
			out->set(i, (high / low) * right);
		}

		return out;
	}

	template <typename T>
	inline NDArray<T>* normal(NDArray<T>* loc, NDArray<T>* scale) {
		loc->assertMatchShape(scale);
		srand(getSeed());
		NDArray<T>* out = loc->emptyLike();

		for (int i = 0; i < loc->size(); i++) {
			T u = loc->get(i);
			T o = scale->get(i);
			T x = rand() / RAND_MAX;
			T o2 = o * o;
			T left = 1 / std::sqrt(2 * pi * o2);
			T right = std::exp(-((std::pow(x - u, 2) / 2 * o2)));
			out->set(i, left * right);
		}

		return out;
	}

	template <typename T>
	inline NDArray<T>* pareto(NDArray<T>* a) {
		// assert all of a is positive
		NDArray<T>* out = a->emptyLike();
		srand(getSeed());

		for (int i = 0; i < out->size(); i++) {
			T x = rand() / RAND_MAX;
			T av = a->get(i);
			out->set(i, av / (std::pow(x, av + 1)));
		}

		return out;
	}

	template <typename T>
	inline NDArray<T>* poisson(NDArray<T>* lam) {
		NDArray<T>* out = lam->emptyLike();
		srand(getSeed());

		for (int i = 0; i < out->size(); i++) {
			T k = rand() / RAND_MAX;
			T lambda = lam->get(i);
			T high = std::pow(lambda, k) * std::exp(-lambda);
			T low = std::tgamma(k);
			out->set(i, high / low);
		}

		return out;
	}

	template <typename T>
	inline NDArray<T>* power(NDArray<T>* a) {
		// assert all of a is positive
		NDArray<T>* out = a->emptyLike();
		srand(getSeed());

		for (int i = 0; i < out->size(); i++) {
			T x = rand() / RAND_MAX;
			T av = a->get(i);
			out->set(i, av * std::pow(x, av - 1));
		}

		return out;
	}

	template <typename T>
	inline NDArray<T>* rayleigh(NDArray<T>* scale) {
		// assert scale >= 0
		NDArray<T>* out = scale->emptyLike();
		srand(getSeed());

		for (int i = 0; i < out->size(); i++) {
			T x = rand() / RAND_MAX;
			T s = scale->get(i);
			T left = x / std::pow(scale, 2);
			T right = std::exp(((-std::pow(x, 2) / 2 * std::pow(scale, 2))));
			out->set(i, left * right);
		}

		return out;
	}

	template <typename T>
	inline NDArray<T>* cauchy(NDArray<T>* x0, NDArray<T>* g) {
		x0->assertMatchShape(g);
		NDArray<T>* out = x0->emptyLike();
		srand(getSeed());

		for (int i = 0; i < x0->size(); i++) {
			T x = rand() / RAND_MAX;
			T xv = x0->get(i);
			T gv = g->get(i);
			T exp = 1 + std::pow(((x - xv) / gv), 2);
			out->set(i, 1 / (pi * g * exp));
		}

		return out;
	}

	template <typename T>
	inline Vector<T>* standardCauchy(int size) {
		Vector<T>* x0 = Cudheart::VectorOps::zeros<T>(size);
		Vector<T>* g = Cudheart::VectorOps::ones<T>(size);
		return cauchy(x0, g);
	}

	template <typename T>
	inline Matrix<T>* standardCauchy(int height, int width) {
		Matrix<T>* x0 = Cudheart::MatrixOps::zeros<T>(height, width);
		Matrix<T>* g = Cudheart::MatrixOps::ones<T>(height, width);
		return cauchy(x0, g);
	}

	template <typename T>
	inline Vector<T>* standardExponantial(int size) {
		Vector<T>* scale = Cudheart::VectorOps::ones<T>(size);
		return exponential(scale);
	}

	template <typename T>
	inline Matrix<T>* standardExponantial(int height, int width) {
		Matrix<T>* scale = Cudheart::MatrixOps::ones<T>(height, width);
		return exponential(scale);
	}

	template <typename T>
	inline NDArray<T>* standardT(NDArray<T>* df) {
		// df > 0
		NDArray<T>* out = df->emptyLike();

		for (int i = 0; i < out->size(); i++) {
			T x = rand() / RAND_MAX;
			T d = df->get(i);
			T high = std::gamma_distribution<>()((d + 1) / 2);
			T low = std::sqrt(pi * d) * std::gamma_distribution<>()(d / 2);
			T left = high / low;
			T base = 1 + (std::pow(x, 2), d);
			T p = -((df + 1) / 2);
			T right = std::pow(base, p);
			out->set(i, left * right);
		}

		return out;
	}

	template <typename T>
	inline Vector<T>* uniform(T low, T high, int size) {
		srand(getSeed());
		Vector<T>* out = new Vector<T>(size);

		for (int i = 0; i < out->size(); i++) {
			T x = rand() / RAND_MAX;
			out->set(i, low + (x * (high - low)));
		}

		return out;
	}

	template <typename T>
	inline Vector<T>* uniform(T high, int size) {
		return uniform<T>(0, high, size);
	}

	template <typename T>
	inline Vector<T>* uniform(int size) {
		return uniform(0, RAND_MAX, size);
	}

	template <typename T>
	inline NDArray<T>* vonmises(NDArray<T>* mu, NDArray<T>* kappa) {
		// kappa > 0
		mu->assertMatchShape(kappa);
		NDArray<T>* out = mu->emptyLike();
		srand(getSeed());

		for (int i = 0; i < out->size(); i++) {
			T x = rand() / RAND_MAX;
			T u = mu->get(i);
			T k = kappa->get(i);
			T high = std::exp(k * std::cos(x - u));
			T low = 2 * pi * std::cyl_bessel_i(0, k);
			out->set(i, high / low);
		}

		return out;
	}

	template <typename T>
	inline NDArray<T>* wald(NDArray<T>* mean, NDArray<T>* scale) {
		mean->assertMatchShape(scale);
		NDArray<T>* out = mean->emptyLike();
		srand(getSeed());

		for (int i = 0; i < out->size(); i++) {
			T x = rand() / RAND_MAX;
			T m = mean->get(i);
			T s = scale->get(i);
			T left = std::sqrt(s / (2 * pi * std::pow(x, 3)));
			T right = std::exp(((-s * std::pow(x - mean, 2)) / (2 * std::pow(scale, 2) * x)));
			out->set(i, m + left * right);
		}

		return out;
	}

	template <typename T>
	inline NDArray<T>* zipf(NDArray<T>* a) {
		NDArray<T>* out = a->emptyLike();
		srand(getSeed());

		for (int i = 0; i < a->size(); i++) {
			T k = rand() / RAND_MAX;
			T av = a->get(i);
			T high = std::pow(k, -av);
			T low = std::riemann_zeta(av);
			out->set(i, high / low);
		}

		return out;
	}
}