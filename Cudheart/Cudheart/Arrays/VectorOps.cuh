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
		return new Vector<T>(arr->getSize());
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
		if (end - start < jump) {
			return new Vector<T>(0);
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
		return full<T>(arr->getSize(), value);
	}

	template <typename T>
	Vector<T>* linspace(T start, T stop, T num, bool endpoint) {
		T jump = (stop - start) / num;
		if (endpoint) {
			jump = (stop - start) / (num - 1);
			stop += jump;
		}
		return arange<T>(start, stop, jump);
	}

	template <typename T>
	Vector<T>* linspace(T start, T stop, T num) {
		return linspace<T>(start, stop, num, true);
	}

	template <typename T>
	Vector<T>* linspace(T start, T stop) {
		return linspace<T>(start, stop, (T)50, true);
	}

	template <typename T>
	Vector<T>* ones(int len) {
		return full<T>(len, 1);
	}

	template <typename T>
	Vector<T>* onesLike(Vector<T>* arr) {
		return ones<T>(arr->getSize());
	}

	template <typename T>
	Vector<T>* zeros(int len) {
		return full<T>(len, 0);
	}

	template <typename T>
	Vector<T>* zerosLike(Vector<T>* arr) {
		return zeros<T>(arr->getSize());
	}

	template <typename T>
	Vector<T>* logspace(T start, T stop, T num, bool endpoint, double base) {
		T jump = (stop - start) / num;
		if (endpoint) {
			jump = (stop - start) / (num - 1);
			stop += jump;
		}
		int len = (int)((end - start) / jump);
		Vector<T>* out = empty<T>(len);

		for (int i = 0; start < end; start += jump) {
			out->set(i++, pow(base, start));
		}

		return out;
	}

	template <typename T>
	Vector<T>* logspace(T start, T stop, T num, bool endpoint) {
		return logspace<T>(start, stop, num, endpoint, 10.0);
	}

	template <typename T>
	Vector<T>* logspace(T start, T stop, T num) {
		return logspace<T>(start, stop, num, true, 10.0);
	}

	template <typename T>
	Vector<T>* logspace(T start, T stop, T num, double base) {
		return logspace<T>(start, stop, num, true, base);
	}

	template <typename T>
	Vector<T>* logspace(T start, T stop, double base) {
		return logspace<T>(start, stop, (T)50, true, base);
	}

	template <typename T>
	Vector<T>* logspace(T start, T stop) {
		return logspace<T>(start, stop, (T)50, true, 10.0);
	}

	template <typename T>
	Vector<T>* geomspace(T start, T stop, T num, bool endpoint) {
		start = log10(start);
		stop = log10(stop);
		auto res = logspace(start, stop, num, endpoint);
		res.set(0, start);
		if (endpoint) {
			res.set(-1, stop);
		}
		return res;
	}

	template <typename T>
	Vector<T>* geomspace(T start, T stop, T num) {
		return geomspace<T>(start, stop, num, true);
	}

	template <typename T>
	Vector<T>* geomspace(T start, T stop) {
		return geomspace<T>(start, stop, (T)50, true);
	}
};