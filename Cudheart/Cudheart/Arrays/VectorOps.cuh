#pragma once

#include "../Util.cuh"
#include "Vector.cuh"
#include "Matrix.cuh"
#include "../Exceptions/Exceptions.cuh"

namespace Cudheart::VectorOps {
	using NDArrays::Vector;
	using NDArrays::Matrix;

	/// <summary>
	/// get an empty vector with the same size as the other vector
	/// </summary>
	/// <typeparam name="T"> - the type of the vector</typeparam>
	/// <param name="arr"> - the reference vector</param>
	/// <returns>the resulting vector</returns>
	template <typename T>
	Vector<T>* emptyLike(Vector<T>* arr) {
		return new Vector<T>(arr->size());
	}

	/// <summary>
	/// create a vector from an arange of values
	/// </summary>
	/// <typeparam name="T"> - the type of the matrix </typeparam>
	/// <param name="start"> - the beginning value of the matrix </param>
	/// <param name="end"> - the end value of the matrix </param>
	/// <param name="jump"> - the difference between each element</param>
	/// <returns>the resulting vector</returns>
	template <typename T>
	Vector<T>* arange(T start, T end, T jump) {
		if (signbit((double)(end - start)) != signbit((double)jump)) {
			ostringstream os;
			os << "cannot define range from " << start << " to " << end;
			BadValueException(os.str());
		}

		// thanks c++
		float diff = (float)end - (float)start;
		float steps = diff / (float)jump;
		int len = (int)ceil(steps);

		Vector<T>* out = new Vector<T>(len);

		for (int i = 0; i < len; start += jump, i++) {
			out->set(i, start);
		}

		return out;
	}

	/// <summary>
	/// create a vector from an arange of values
	/// </summary>
	/// <typeparam name="T"> - the type of the matrix </typeparam>
	/// <param name="start"> - the beginning value of the matrix </param>
	/// <param name="end"> - the end value of the matrix </param>
	/// <returns>the resulting vector</returns>
	template <typename T>
	Vector<T>* arange(T end, T jump) {
		return arange<T>((T)0, end, jump);
	}

	template <typename T>
	Vector<T>* arange(T end) {
		return arange<T>((T)0, end, (T)1);
	}

	template <typename T>
	Vector<T>* full(int s, T value) {
		Vector<T>* out = new Vector<T>(s);

		for (int i = 0; i < s; i++) {
			out->set(i, value);
		}

		return out;
	}

	template <typename T>
	Vector<T>* fullLike(Vector<T>* arr, T value) {
		return full<T>(arr->size(), value);
	}

	template <typename T>
	Vector<T>* linspace(T start, T stop, int num, bool endpoint) {
		double diff = (double)stop - (double)start;

		Vector<T>* vec = new Vector<T>(num);

		if (endpoint) {
			num--;
			vec->set(-1, stop);
		}

		T jump = (T)(diff / num);

		for (int i = 0; i < num; i++) {
			vec->set(i, (T)(start + jump * i));
		}

		return vec;
	}

	template <typename T>
	Vector<T>* linspace(T start, T stop, int num) {
		return linspace<T>(start, stop, num, true);
	}

	template <typename T>
	Vector<T>* linspace(T start, T stop) {
		return linspace<T>(start, stop, 50, true);
	}

	template <typename T>
	Vector<T>* ones(int len) {
		return full<T>(len, 1);
	}

	template <typename T>
	Vector<T>* onesLike(Vector<T>* arr) {
		return ones<T>(arr->size());
	}

	template <typename T>
	Vector<T>* zeros(int len) {
		return full<T>(len, 0);
	}

	template <typename T>
	Vector<T>* zerosLike(Vector<T>* arr) {
		return zeros<T>(arr->size());
	}

	template <typename T>
	Vector<T>* logspace(T start, T stop, int num, bool endpoint, T base) {
		double diff = (double)stop - (double)start;

		Vector<T>* vec = new Vector<T>(num);

		if (endpoint) {
			num--;
			vec->set(-1, pow(base, stop));
		}

		T jump = (T)(diff / num);

		for (int i = 0; i < num; i++) {
			vec->set(i, pow(base, (T)(start + jump * i)));
		}

		return vec;
	}

	template <typename T>
	Vector<T>* logspace(T start, T stop, int num, bool endpoint) {
		return logspace<T>(start, stop, num, endpoint, 10.0);
	}

	template <typename T>
	Vector<T>* logspace(T start, T stop, int num, double base) {
		return logspace<T>(start, stop, num, true, base);
	}

	template <typename T>
	Vector<T>* logspace(T start, T stop, int num) {
		return logspace<T>(start, stop, num, true, 10.0);
	}

	template <typename T>
	Vector<T>* logspace(T start, T stop) {
		return logspace<T>(start, stop, 50, true, 10.0);
	}

	template <typename T>
	Vector<T>* geomspace(T start, T stop, int num, bool endpoint) {
		return logspace((T)log10(start), (T)log10(stop), num, endpoint, (T)10.0);
	}

	template <typename T>
	Vector<T>* geomspace(T start, T stop, int num) {
		return geomspace<T>(start, stop, num, true);
	}

	template <typename T>
	Vector<T>* geomspace(T start, T stop) {
		return geomspace<T>(start, stop, (T)50, true);
	}
};