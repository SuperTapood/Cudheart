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

namespace Cudheart {
	namespace MatrixOps {
#pragma region vector_matrix
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
			Matrix<T>* out = new Matrix<T>(len, a.getSize());

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
			return fromVector(VectorOps::arange<T>(start, end, jump), height, width, true);
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
		/// get a new matrix like the given one
		/// </summary>
		/// <typeparam name="T"> - the type of the matrix</typeparam>
		/// <param name="mat"> - the reference matrix</param>
		/// <returns>a new empty matrix with matching dims</returns>
		template <typename T>
		Matrix<T>* emptyLike(Matrix<T>* mat) {
			return new Matrix<T>(mat->getWidth(), mat->getHeight());
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
			return fromVector<T>(VectorOps::full(width * height, value), height, width, true);
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
			return ones<T>(mat->getWidth(), mat->getHeight());
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
			return zeros<T>(mat->getWidth(), mat->getHeight());
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

#pragma region eye
		/// <summary>
		/// get a matrix with ones on its diagonals
		/// </summary>
		/// <typeparam name="T"> - the type of the matrix</typeparam>
		/// <param name="N"> - the number of rows in the output</param>
		/// <param name="M"> - the number of columns in the output</param>
		/// <param name="k"> - index of the diagonal</param>
		/// <returns>a matrix where all elements are 0 except for the k-th diagonal, whose values are equal to 1</returns>
		template <typename T>
		Matrix<T>* eye(int N, int M, int k) {
			Matrix<T>* mat = zeros<T>(N, M);

			for (int i = 0, j = k; i < N && j < M; i++, j++) {
				j %= M;
				mat->set(i, j, (T)1);
			}

			return mat;
		}

		/// <summary>
		/// get a matrix with ones on its diagonals
		/// </summary>
		/// <typeparam name="T"> - the type of the matrix</typeparam>
		/// <param name="N"> - the number of rows in the output</param>
		/// <param name="k"> - index of the diagonal</param>
		/// <returns>a matrix where all elements are 0 except for the k-th diagonal, whose values are equal to 1</returns>
		template <typename T>
		Matrix<T>* eye(int N, int k) {
			return eye<T>(N, N, k);
		}

		/// <summary>
		/// get a matrix with ones on its diagonals
		/// </summary>
		/// <typeparam name="T"> - the type of the matrix</typeparam>
		/// <param name="N"> - the number of rows in the output</param>
		/// <returns>a matrix where all elements are 0 except for the diagonal, whose values are equal to 1</returns>
		template <typename T>
		Matrix<T>* eye(int N) {
			return eye<T>(N, N, 0);
		}
#pragma endregion

		/// <summary>
		/// return coordinate matrices for vectorized evaluations of given one dimensional coordinate arrays a and b
		/// </summary>
		/// <typeparam name="T"> - the type of the output matrix</typeparam>
		/// <typeparam name="U"> - the type of the a input vector</typeparam>
		/// <typeparam name="K"> - the type of the b input vector</typeparam>
		/// <param name="a"> - first input vector</param>
		/// <param name="b"> - second input vector</param>
		/// <returns>an array containing two matrices</returns>
		template <typename T, typename U, typename K>
		Matrix<T>** meshgrid(Vector<U>* a, Vector<K>* b) {
			Matrix<T>** out = new Matrix<T>*[2];
			Matrix<T>* first = new Matrix<T>(b->getSize(), a->getSize());
			Matrix<T>* second = new Matrix<T>(b->getSize(), a->getSize());

			out[0] = first;
			out[1] = second;

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

		template <typename T, typename U, typename K>
		Matrix<T>** meshgrid(Matrix<U>* a, Matrix<K>* b) {
			return meshgrid<T>(a->flatten(), b->flatten());
		}

#pragma region diags

		/// <summary>
		/// construct a diagonal vector
		/// </summary>
		/// <typeparam name="T"> - the type of the output vector</typeparam>
		/// <param name="mat"> - the matrix to construct a diagonal vector from</param>
		/// <param name="k"> - the diagonal in question</param>
		/// <returns>the constructed diagonal vector</returns>
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

			Vector<T>* out = new Vector<T>(len);
			for (int i = 0, j = k; i < mat->getHeight() && j < mat->getWidth(); i++, j++) {
				out->set(i, mat->get(i, j));
			}

			return out;
		}

		/// <summary>
		/// construct a diagonal vector
		/// </summary>
		/// <typeparam name="T"> - the type of the output vector</typeparam>
		/// <param name="mat"> - the matrix to construct a diagonal vector from</param>
		/// <returns>the constructed diagonal vector</returns>
		template <typename T>
		Vector<T>* diag(Matrix<T>* mat) {
			return diag<T>(mat, 0);
		}

		/// <summary>
		/// create a two-dimensional array with the given vector as a diagonal
		/// </summary>
		/// <typeparam name="T"> - the type of the output matrix</typeparam>
		/// <param name="vec"> - the given vector</param>
		/// <param name="k"> - the diagonal to set</param>
		/// <returns>the output matrix</returns>
		template <typename T>
		Matrix<T>* diagflat(Vector<T>* vec, int k) {
			Matrix<T>* mat = zeros<T>(vec->getSize() + k, vec->getSize() + k);

			for (int i = 0, j = k; i < mat->getHeight() && j < mat->getWidth(); i++, j++) {
				mat->set(i, j, vec->get(i));
			}

			return mat;
		}

		/// <summary>
		/// create a two-dimensional array with the given vector as a diagonal
		/// </summary>
		/// <typeparam name="T"> - the type of the output matrix</typeparam>
		/// <param name="vec"> - the given vector</param>
		/// <returns>the output matrix</returns>
		template <typename T>
		Matrix<T>* diagflat(Vector<T>* vec) {
			return diagflat(vec, 0);
		}
#pragma endregion

#pragma region triags

		/// <summary>
		/// a matrix with ones at and below the given diagonal
		/// </summary>
		/// <typeparam name="T"> - the type of the matrix</typeparam>
		/// <param name="N"> - number of rows</param>
		/// <param name="M"> - number of columns</param>
		/// <param name="k"> - the sub diagonal at and below which the matrix is filled</param>
		/// <returns>array with its lower triangle filled with ones and zeros elsewhere</returns>
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

		/// <summary>
		/// a matrix with ones at and below the given diagonal
		/// </summary>
		/// <typeparam name="T"> - the type of the matrix</typeparam>
		/// <param name="N"> - number of rows and columns</param>
		/// <param name="k"> - the sub diagonal at and below which the matrix is filled</param>
		/// <returns>array with its lower triangle filled with ones and zeros elsewhere</returns>
		template <typename T>
		Matrix<T>* tri(int N, int k) {
			return tri<T>(N, N, k);
		}

		/// <summary>
		/// a matrix with ones at and below the given diagonal
		/// </summary>
		/// <typeparam name="T"> - the type of the matrix</typeparam>
		/// <param name="N"> - number of rows and columns</param>
		/// <returns>array with its lower triangle filled with ones and zeros elsewhere</returns>
		template <typename T>
		Matrix<T>* tri(int N) {
			return tri<T>(N, N, 0);
		}

		/// <summary>
		/// return a copy of the matrix with elements above the k-th diagonal zeroed
		/// </summary>
		/// <typeparam name="T"> - the type of the output matrix</typeparam>
		/// <param name="mat"> - the matrix to work on</param>
		/// <param name="k"> - the index of the diagonal</param>
		/// <returns>lower triangle of mat</returns>
		template <typename T>
		Matrix<T>* tril(Matrix<T>* mat, int k) {
			Matrix<T>* out = (Matrix<T>*)mat->copy();

			for (int i = 0; i < mat->getHeight(); i++) {
				for (int j = k + i + 1; j < mat->getWidth(); j++) {
					out->set(i, j, 0);
				}
			}

			return out;
		}

		/// <summary>
		/// return a copy of the matrix with elements above the main diagonal zeroed
		/// </summary>
		/// <typeparam name="T"> - the type of the output matrix</typeparam>
		/// <param name="mat"> - the matrix to work on</param>
		/// <returns>lower triangle of mat</returns>
		template <typename T>
		Matrix<T>* tril(Matrix<T>* mat) {
			return tril(mat, 0);
		}

		/// <summary>
		/// return a copy of the matrix with elements below the k-th diagonal zeroed
		/// </summary>
		/// <typeparam name="T"> - the type of the output matrix</typeparam>
		/// <param name="mat"> - the matrix to work on</param>
		/// <param name="k"> - the index of the diagonal</param>
		/// <returns>upper triangle of mat</returns>
		template <typename T>
		Matrix<T>* triu(Matrix<T>* mat, int k) {
			Matrix<T>* out = zerosLike<T>(mat);

			for (int i = 0; i < mat->getHeight(); i++) {
				for (int j = k + i; j < mat->getWidth(); j++) {
					out->set(i, j, mat->get(i, j));
				}
			}

			return out;
		}

		/// <summary>
		/// return a copy of the matrix with elements below the main diagonal zeroed
		/// </summary>
		/// <typeparam name="T"> - the type of the output matrix</typeparam>
		/// <param name="mat"> - the matrix to work on</param>
		/// <returns>upper triangle of mat</returns>
		template <typename T>
		Matrix<T>* triu(Matrix<T>* mat) {
			return triu(mat, 0);
		}

		/// <summary>
		/// generate a Vandermonde matrix
		/// </summary>
		/// <typeparam name="T"> - the type of the matrix</typeparam>
		/// <param name="vec"> - a vector input array</param>
		/// <param name="N"> - number of columns in the output</param>
		/// <param name="increasing"> - whether or not to increase powers from left to right or the other way around</param>
		/// <returns>output vandermonde matrix</returns>
		template <typename T>
		Matrix<T>* vander(Vector<T>* vec, int N, bool increasing) {
			Matrix<T>* out = new Matrix<T>(vec->getSize(), N);

			for (int i = 0; i < out->getHeight(); i++) {
				for (int j = 0; j < out->getWidth(); j++) {
					out->set(i, j, pow(vec->get(i), j));
				}
			}


			if (!increasing) {
				out->reverseRows(true);

				// out->reverseRows(true);
			}

			

			

			// out = out->rot90(3, true);

			/*if (increasing) {
				return out->flip(true);
			}*/

			return out;
		}

		/// <summary>
		/// generate a Vandermonde matrix
		/// </summary>
		/// <typeparam name="T"> - the type of the matrix</typeparam>
		/// <param name="vec"> - a vector input array</param>
		/// <param name="N"> - number of columns in the output</param>
		/// <returns>output vandermonde matrix</returns>
		template <typename T>
		Matrix<T>* vander(Vector<T>* vec, int N) {
			return vander(vec, N, false);
		}

		/// <summary>
		/// generate a Vandermonde matrix
		/// </summary>
		/// <typeparam name="T"> - the type of the matrix</typeparam>
		/// <param name="vec"> - a vector input array</param>
		/// <param name="increasing"> - whether or not to increase powers from left to right or the other way around</param>
		/// <returns>output vandermonde matrix</returns>
		template <typename T>
		Matrix<T>* vander(Vector<T>* vec, bool increasing) {
			return vander(vec, vec->getSize(), increasing);
		}

		/// <summary>
		/// generate a Vandermonde matrix
		/// </summary>
		/// <typeparam name="T"> - the type of the matrix</typeparam>
		/// <param name="vec"> - a vector input array</param>
		/// <returns>output vandermonde matrix</returns>
		template <typename T>
		Matrix<T>* vander(Vector<T>* vec) {
			return vander(vec, vec->getSize(), false);
		}

#pragma endregion
	};
}