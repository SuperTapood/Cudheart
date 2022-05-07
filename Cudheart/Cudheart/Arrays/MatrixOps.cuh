#pragma once

#include "../Util.cuh"
#include "Matrix.cuh"
#include "Vector.cuh"
#include "VectorOps.cuh"
#include "../Exceptions/Exceptions.cuh"

using Cudheart::NDArrays::Matrix;
using Cudheart::NDArrays::Vector;
using namespace Cudheart::Exceptions;

// all this namespace does is call VectorOps functions and cast the resulting vector to a matrix lol

namespace Cudheart::MatrixOps {
#pragma region vector_matrix
	/// <summary>
	/// convert an array to a matrix
	/// </summary>
	/// <typeparam name="T"> - the type of the given array</typeparam>
	/// <param name="arr"> - the array to convert to a matrix</param>
	/// <param name="height"> - the desired height of the matrix</param>
	/// <param name="width"> - the desired width of the matrix</param>
	/// <returns>a matrix with the given parameters</returns>
	template <typename T>
	Matrix<T>* asMatrix(T* arr, int height, int width) {
		return new Matrix<T>(arr, height, width);
	}

	/// <summary>
	/// convert a vector to a matrix
	/// </summary>
	/// <typeparam name="T"> - the type of the vector</typeparam>
	/// <param name="vec"> - the vector to deflat</param>
	/// <param name="height"> - the height of the matrix</param>
	/// <param name="width"> - the width of the matrix</param>
	/// <param name="destroy"> - whether or not to destroy the vector when done</param>
	/// <returns>the resulting matrix</returns>
	template <typename T>
	Matrix<T>* fromVector(Vector<T>* vec, int height, int width, bool destroy) {
		if (width * height != vec->getSize()) {
			BadValueException("fromVector",
				"width * height = " + std::to_string(width * height), "width * height = " + std::to_string(vec->getSize()));
		}
		Matrix<T>* res = new Matrix<T>(height, width);

		for (int i = 0; i < res->getSize(); i++) {
			res->set(i, vec->get(i));
		}

		if (destroy) {
			delete vec;
		}
		return res;
	}

	/// <summary>
	/// convert a vector to a matrix
	/// </summary>
	/// <typeparam name="T"> - the type of the matrix</typeparam>
	/// <param name="vec"> - the vector to convert</param>
	/// <param name="height"> - the height of the matrix</param>
	/// <param name="width"> - the width of the matrix</param>
	/// <returns>a matrix with the same values as the vector</returns>
	template <typename T>
	Matrix<T>* fromVector(Vector<T>* vec, int height, int width) {
		if (width * height != vec->getSize()) {
			Exceptions::MatrixConversionException(width, height, vec->getSize()).raise();
		}
		Matrix<T>* out = empty<T>(height, width);

		for (int i = 0; i < out->getSize(); i++) {
			out->set(i, vec->get(i));
		}

		return out;
	}

	/// <summary>
	/// convert a vector array to a matrix.
	/// don't get cute and give this a vector pointer
	/// or cpp will actually send you to the shadow realm jimbo
	/// </summary>
	/// <typeparam name="T"> - the type of the vector array</typeparam>
	/// <param name="vecs"> - the array of vectors to convert</param>
	/// <param name="len"> - the length of the array of vectors</param>
	/// <returns></returns>
	template <typename T>
	Matrix<T>* fromVectorArray(Vector<T>* vecs, int len) {
		Vector<T> a = vecs[0];
		Matrix<T>* out = empty<T>(len, a.getSize());

		for (int i = 0; i < len; i++) {
			for (int j = 0; j < a.getSize(); j++) {
				a->assertMatchSize(vecs[i].getSize());
				Vector<T> vec = vecs[i];
				out->set(i, j, vec.get(j));
			}
		}

		return out;
	}
#pragma endregion

#pragma region arange
	/// <summary>
	/// create a matrix from an arange of values
	/// </summary>
	/// <typeparam name="T"> - the type of the matrix </typeparam>
	/// <param name="start"> - the beginning value of the matrix </param>
	/// <param name="end"> - the end value of the matrix </param>
	/// <param name="jump"> - the difference between each element</param>
	/// <param name="height"> - the height of the element</param>
	/// <param name="width"> - the width of the element</param>
	/// <returns>the resulting matrix</returns>
	template <typename T>
	Matrix<T>* arange(T start, T end, T jump, int height, int width) {
		return fromVector(VectorOps::arange<T>(start, end, jump), width, height, true);
	}

	/// <summary>
	/// create a matrix from an arange of values
	/// </summary>
	/// <typeparam name="T"> - the type of the matrix </typeparam>
	/// <param name="end"> - the end value of the matrix </param>
	/// <param name="jump"> - the difference between each element</param>
	/// <param name="height"> - the height of the element</param>
	/// <param name="width"> - the width of the element</param>
	/// <returns>the resulting matrix</returns>
	template <typename T>
	Matrix<T>* arange(T end, T jump, int height, int width) {
		return arange((T)0, end, jump, height, width);
	}

	/// <summary>
	/// create a matrix from an arange of values
	/// </summary>
	/// <typeparam name="T"> - the type of the matrix </typeparam>
	/// <param name="end"> - the end value of the matrix </param>
	/// <param name="height"> - the height of the element</param>
	/// <param name="width"> - the width of the element</param>
	/// <returns>the resulting matrix</returns>
	template <typename T>
	Matrix<T>* arange(T end, int height, int width) {
		return arange((T)0, end, (T)1, height, width);
	}
#pragma endregion

#pragma region fulls
	/// <summary>
	/// get a new empty matrix
	/// </summary>
	/// <typeparam name="T"> - the type of the matrix</typeparam>
	/// <param name="width"> - the width of the matrix</param>
	/// <param name="height"> - the height of the matrix</param>
	/// <returns>a new empty matrix</returns>
	template <typename T>
	Matrix<T>* empty(int height, int width) {
		return new Matrix<T>(height, width);
	}

	/// <summary>
	/// get a new matrix like the given one
	/// </summary>
	/// <typeparam name="T"> - the type of the matrix</typeparam>
	/// <param name="mat"> - the reference matrix</param>
	/// <returns>a new empty matrix with matching dims</returns>
	template <typename T>
	Matrix<T>* emptyLike(Matrix<T>* mat) {
		return empty<T>(mat->getWidth(), mat->getHeight());
	}

	/// <summary>
	/// get a full matrix with the given value
	/// </summary>
	/// <typeparam name="T"> - the type of the matrix</typeparam>
	/// <param name="height"> - the height of the matrix</param>
	/// <param name="width"> - the width of the matrix</param>
	/// <param name="value"> - the value to fill the matrix with</param>
	/// <returns>a full matrix</returns>
	template <typename T>
	Matrix<T>* full(int height, int width, T value) {
		return fromVector<T>(VectorOps::full(width * height, value), width, height, true);
	}

	/// <summary>
	/// get a full matrix with the given value with the same dims as the given matrix
	/// </summary>
	/// <typeparam name="T"> - the type of the matrix</typeparam>
	/// <param name="mat"> - the reference matrix</param>
	/// <param name="value"> - the value to fill with</param>
	/// <returns>a full matrix like the given matrix</returns>
	template <typename T>
	Matrix<T>* fullLike(Matrix<T>* mat, T value) {
		return full<T>(mat->getWidth(), mat->getHeight(), value);
	}

	/// <summary>
	/// get a matrix filled with ones
	/// </summary>
	/// <typeparam name="T"> - the type of the matrix</typeparam>
	/// <param name="height"> - the height of the matrix</param>
	/// <param name="width"> - the width of the matrix</param>
	/// <returns>a matrix filled with ones</returns>
	template <typename T>
	Matrix<T>* ones(int height, int width) {
		return full(height, width, 1);
	}

	/// <summary>
	/// a matrix filled with ones with the same dims as the given matrix
	/// </summary>
	/// <typeparam name="T"> - the type of the matrix</typeparam>
	/// <param name="mat"> - the reference matrix</param>
	/// <returns>a matrix filled with ones</returns>
	template <typename T>
	Matrix<T>* onesLike(Matrix<T>* mat) {
		return ones(mat->getWidth(), mat->getHeight());
	}

	/// <summary>
	/// get a matrix filled with zeros
	/// </summary>
	/// <typeparam name="T"> - the type of the matrix</typeparam>
	/// <param name="height"> - the height of the matrix</param>
	/// <param name="width"> - the width of the matrix</param>
	/// <returns>a matrix filled with zeros</returns>
	template <typename T>
	Matrix<T>* zeros(int height, int width) {
		return full<T>(height, width, 0);
	}

	/// <summary>
	/// get a matrix filled with zeros with the same dims as the given matrix
	/// </summary>
	/// <typeparam name="T"> - the type of the matrix</typeparam>
	/// <param name="mat"> - the reference matrix</param>
	/// <returns>a matrix filled with zeros</returns>
	template <typename T>
	Matrix<T>* zerosLike(Matrix<T>* mat) {
		return zeros(mat->getWidth(), mat->getHeight());
	}
#pragma endregion 
#pragma region linspace
	/// <summary>
	/// get a matrix made out of linearly spaced values
	/// </summary>
	/// <typeparam name="T"> - the type of the matrix</typeparam>
	/// <param name="start"> - the start value</param>
	/// <param name="stop"> - the final value</param>
	/// <param name="num"> - the number of steps to take </param>
	/// <param name="endpoint"> - whether or not to include the final value in the matrix</param>
	/// <param name="height"> - the height of the matrix</param>
	/// <param name="width"> - the width of the matrix</param>
	/// <returns>a matrix filled linearly spaced values</returns>
	template <typename T>
	Matrix<T>* linspace(T start, T stop, T num, bool endpoint, int height, int width) {
		return fromVector(VectorOps::linspace<T>(start, stop, num, endpoint), height, width, true);
	}

	/// <summary>
	/// get a matrix made out of linearly spaced values
	/// </summary>
	/// <typeparam name="T"> - the type of the matrix</typeparam>
	/// <param name="start"> - the start value</param>
	/// <param name="stop"> - the final value</param>
	/// <param name="num"> - the number of steps to take </param>
	/// <param name="height"> - the height of the matrix</param>
	/// <param name="width"> - the width of the matrix</param>
	/// <returns>a matrix filled linearly spaced values</returns>
	template <typename T>
	Matrix<T>* linspace(T start, T stop, T num, int height, int width) {
		return linspace(start, stop, num, true, height, width);
	}

	/// <summary>
	/// get a matrix made out of linearly spaced values
	/// </summary>
	/// <typeparam name="T"> - the type of the matrix</typeparam>
	/// <param name="start"> - the start value</param>
	/// <param name="stop"> - the final value</param>
	/// <param name="height"> - the height of the matrix</param>
	/// <param name="width"> - the width of the matrix</param>
	/// <returns>a matrix filled linearly spaced values</returns>
	template <typename T>
	Matrix<T>* linspace(T start, T stop, int height, int width) {
		return linspace(start, stop, (T)50, true, height, width);
	}

#pragma endregion

#pragma region logspace
	/// <summary>
	/// get a matrix with logarithmically spaced values
	/// </summary>
	/// <typeparam name="T"> - the type of the matrix</typeparam>
	/// <param name="start"> - the starting value of the matrix</param>
	/// <param name="stop"> - the final value of the matrix</param>
	/// <param name="num"> - the number of steps to take</param>
	/// <param name="endpoint"> - whether or not to include the final value</param>
	/// <param name="base"> - the base to raise the values to</param>
	/// <param name="height"> - the height of the matrix</param>
	/// <param name="width"> - the width of the matrix</param>
	/// <returns>a logarithmically spaced matrix</returns>
	template <typename T>
	Matrix<T>* logspace(T start, T stop, T num, bool endpoint, double base, int height, int width) {
		return fromVector(VectorOps::logspace<T>(start, stop, num, endpoint, base), height, width, true);
	}

	/// <summary>
	/// get a matrix with logarithmically spaced values
	/// </summary>
	/// <typeparam name="T"> - the type of the matrix</typeparam>
	/// <param name="start"> - the starting value of the matrix</param>
	/// <param name="stop"> - the final value of the matrix</param>
	/// <param name="num"> - the number of steps to take</param>
	/// <param name="endpoint"> - whether or not to include the final value</param>
	/// <param name="height"> - the height of the matrix</param>
	/// <param name="width"> - the width of the matrix</param>
	/// <returns>a logarithmically spaced matrix</returns>
	template <typename T>
	Matrix<T>* logspace(T start, T stop, T num, bool endpoint, int height, int width) {
		return logspace<T>(start, stop, num, endpoint, 10.0, height, width);
	}

	/// <summary>
	/// get a matrix with logarithmically spaced values
	/// </summary>
	/// <typeparam name="T"> - the type of the matrix</typeparam>
	/// <param name="start"> - the starting value of the matrix</param>
	/// <param name="stop"> - the final value of the matrix</param>
	/// <param name="num"> - the number of steps to take</param>
	/// <param name="height"> - the height of the matrix</param>
	/// <param name="width"> - the width of the matrix</param>
	/// <returns>a logarithmically spaced matrix</returns>
	template <typename T>
	Matrix<T>* logspace(T start, T stop, T num, int height, int width) {
		return logspace<T>(start, stop, num, true, 10.0, height, width);
	}

	/// <summary>
	/// get a matrix with logarithmically spaced values
	/// </summary>
	/// <typeparam name="T"> - the type of the matrix</typeparam>
	/// <param name="start"> - the starting value of the matrix</param>
	/// <param name="stop"> - the final value of the matrix</param>
	/// <param name="num"> - the number of steps to take</param>
	/// <param name="base"> - the base to raise the values to</param>
	/// <param name="height"> - the height of the matrix</param>
	/// <param name="width"> - the width of the matrix</param>
	/// <returns>a logarithmically spaced matrix</returns>
	template <typename T>
	Matrix<T>* logspace(T start, T stop, T num, double base, int height, int width) {
		return logspace<T>(start, stop, num, true, base, height, width);
	}

	/// <summary>
	/// get a matrix with logarithmically spaced values
	/// </summary>
	/// <typeparam name="T"> - the type of the matrix</typeparam>
	/// <param name="start"> - the starting value of the matrix</param>
	/// <param name="stop"> - the final value of the matrix</param>
	/// <param name="base"> - the base to raise the values to</param>
	/// <param name="height"> - the height of the matrix</param>
	/// <param name="width"> - the width of the matrix</param>
	/// <returns>a logarithmically spaced matrix</returns>
	template <typename T>
	Matrix<T>* logspace(T start, T stop, double base, int height, int width) {
		return logspace<T>(start, stop, (T)50, true, base, height, width);
	}

	/// <summary>
	/// get a matrix with logarithmically spaced values
	/// </summary>
	/// <typeparam name="T"> - the type of the matrix</typeparam>
	/// <param name="start"> - the starting value of the matrix</param>
	/// <param name="stop"> - the final value of the matrix</param>
	/// <param name="height"> - the height of the matrix</param>
	/// <param name="width"> - the width of the matrix</param>
	/// <returns>a logarithmically spaced matrix</returns>
	template <typename T>
	Matrix<T>* logspace(T start, T stop, int height, int width) {
		return logspace<T>(start, stop, (T)50, true, 10.0, height, width);
	}
#pragma endregion

#pragma region geomspace
	/// <summary>
	/// get a matrix with geometrically spaced values
	/// </summary>
	/// <typeparam name="T"> - the type of the matrix</typeparam>
	/// <param name="start"> - the starting value</param>
	/// <param name="stop"> - the final value</param>
	/// <param name="num"> - the number of steps between start and stop</param>
	/// <param name="endpoint"> - whether or not to include the final value</param>
	/// <param name="height"> - the height of the matrix</param>
	/// <param name="width"> - the width of the matrix</param>
	/// <returns>a geometrically spaced values matrix</returns>
	template <typename T>
	Matrix<T>* geomspace(T start, T stop, T num, bool endpoint, int height, int width) {
		return fromVector(VectorOps::geomspace(start, stop, num, endpoint), height, width, true);
	}

	/// <summary>
	/// get a matrix with geometrically spaced values
	/// </summary>
	/// <typeparam name="T"> - the type of the matrix</typeparam>
	/// <param name="start"> - the starting value</param>
	/// <param name="stop"> - the final value</param>
	/// <param name="num"> - the number of steps between start and stop</param>
	/// <param name="height"> - the height of the matrix</param>
	/// <param name="width"> - the width of the matrix</param>
	/// <returns>a geometrically spaced values matrix</returns>
	template <typename T>
	Matrix<T>* geomspace(T start, T stop, T num, int height, int width) {
		return geomspace<T>(start, stop, num, true, height, width);
	}

	/// <summary>
	/// get a matrix with geometrically spaced values
	/// </summary>
	/// <typeparam name="T"> - the type of the matrix</typeparam>
	/// <param name="start"> - the starting value</param>
	/// <param name="stop"> - the final value</param>
	/// <param name="height"> - the height of the matrix</param>
	/// <param name="width"> - the width of the matrix</param>
	/// <returns>a geometrically spaced values matrix</returns>
	template <typename T>
	Matrix<T>* geomspace(T start, T stop, int height, int width) {
		return geomspace<T>(start, stop, (T)50, true, height, width);
	}
#pragma endregion

	template <typename T>
	Matrix<T>* eye(int N, int M, int k) {
		Matrix<T>* mat = zeros<T>(N, M);

		for (int i = 0, j = k; i < N && j < M; i++, j++) {
			j %= M;
			mat->set(i, j, (T)1);
		}

		return mat;
	}

	template <typename T>
	Matrix<T>* eye(int N, int k) {
		return eye<T>(N, N, k);
	}

	template <typename T>
	Matrix<T>* eye(int N) {
		return eye<T>(N, N, 0);
	}

	template <typename T>
	Matrix<T>* identity(int N) {
		return eye<T>(N, N, 0);
	}

	template <typename T, typename U, typename K>
	Matrix<T>* meshgrid(Vector<U>* a, Vector<K>* b) {
		Matrix<T>* out = (Matrix<T>*)malloc(sizeof(Matrix<T>) * 2);
		Matrix<T>* first = empty<T>(b->getSize(), a->getSize());
		Matrix<T>* second = empty<T>(b->getSize(), a->getSize());
		out[0] = *first;
		out[1] = *second;

		for (int i = 0; i < b->getSize(); i++) {
			for (int j = 0; j < a->getSize(); j++) {
				first->set(i, j, a->get(j));
			}
		}

		for (int i = 0; i < b->getSize(); i++) {
			for (int j = 0; j < a->getSize(); j++) {
				second->set(i, j, b->get(i));
			}
		}

		return out;
	}

	template <typename T>
	Vector<T>* diag(Matrix<T>* mat, int k) {
		int len;
		if (mat->getWidth() > mat->getHeight()) {
			len = mat->getHeight();
		}
		else {
			len = mat->getWidth();
		}

		len = len - k;

		Vector<T>* out = VectorOps::empty<T>(len);
		for (int i = 0, j = k; i < mat->getHeight() && j < mat->getWidth(); i++, j++) {
			out->set(i, mat->get(i, j));
		}

		return out;
	}

	template <typename T>
	Matrix<T>* diagflat(Vector<T>* vec, int k) {
		Matrix<T>* mat = zeros<T>(vec->getSize() + k, vec->getSize() + k);

		for (int i = 0, j = k; i < mat->getHeight() && j < mat->getWidth(); i++, j++) {
			mat->set(i, j, vec->get(i));
		}

		return mat;
	}

	template <typename T>
	Matrix<T>* diagflat(Vector<T>* vec) {
		return diagflat(vec, 0);
	}

	template <typename T>
	Matrix<T>* tri(int N, int M, int k) {
		Matrix<T>* out = zeros<T>(N, M);

		for (int i = 0; i < N; i++) {
			for (int j = 0; j <= k + i && j < M; j++) {
				out->set(i, j, 1);
			}
		}

		return out;
	}

	template <typename T>
	Matrix<T>* tri(int N, int k) {
		return tri<T>(N, N, k);
	}

	template <typename T>
	Matrix<T>* tri(int N) {
		return tri<T>(N, N, 0);
	}

	template <typename T>
	Matrix<T>* tril(Matrix<T>* mat, int k) {
		Matrix<T>* out = zerosLike<T>(mat);

		for (int i = 0; i < mat->getHeight(); i++) {
			for (int j = mat->getWidth(); j > i + k; j++) {
				out->set(i, j, mat->get(i, j));
			}
		}

		return out;
	}

	template <typename T>
	Matrix<T>* tril(Matrix<T>* mat) {
		return tril(mat, 0);
	}

	template <typename T>
	Matrix<T>* triu(Matrix<T>* mat, int k) {
		// the algorithm
		// Matrix<T>* a = transpose(mat);
		// Matrix<T>* b = tril(a);
		// Matrix<T>* c = transpose(b);
		// delete a;
		// delete b;
		// return c;

		Matrix<T>* out = zerosLike<T>(mat);

		for (int i = 1 - k; i < mat->getHeight(); i--) {
			for (int j = i; j < mat->getWidth(); j++) {
				out->set(i, j, mat->get(i, j));
			}
		}

		return out;
	}

	template <typename T>
	Matrix<T>* triu(Matrix<T>* mat) {
		return triu(mat, 0);
	}

	template <typename T>
	Matrix<T>* vander(Vector<T>* vec, int N, bool increasing) {
		Matrix<T>* out = empty(vec->getSize(), N);

		for (int i = 0; i < out->getHeight(); i++) {
			for (int j = 0; j < out->getWidth(); j++) {
				out->set(i, j, pow(vec->get(j), i));
			}
		}

		out = out->rotate(-90, true);

		if (increasing) {
			return out->flip(true);
		}

		return out;
	}

	template <typename T>
	Matrix<T>* vander(Vector<T>* vec, int N) {
		return vander(vec, N, false);
	}

	template <typename T>
	Matrix<T>* vander(Vector<T>* vec, bool increasing) {
		return vander(vec, vec->getSize(), increasing);
	}

	template <typename T>
	Matrix<T>* vander(Vector<T>* vec) {
		return vander(vec, vec->getSize(), false);
	}

	template <typename T>
	Matrix<T>* transpose(Matrix<T>* mat) {
		return mat->transpose();
	}
};