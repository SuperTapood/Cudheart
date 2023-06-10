#pragma once

#include "../../Arrays/NDArray.cuh"
#include "../../Internal/Internal.cuh"

#include "../BaseMath/BaseMath.cuh"

#include <fmt/core.h>

namespace CudheartNew::CPP::Math::Linalg {
	template <typename A, typename B, typename T = promote(A, B)>
	T cumDot(NDArray<A>* a, NDArray<B>* b) {
		auto casted = broadcast({ a, b });

		a = (NDArray<A>*)casted[0];
		b = (NDArray<B>*)casted[1];

		T result = (T)0;

		for (int i = 0; i < a->size(); i++) {
			result += a->at(i) * b->at(i);
		}

		return result;
	}

	template <typename A, typename B, typename T = promote(A, B)>
	NDArray<T>* dot(NDArray<A>* a, NDArray<B>* b) {
		Shape resShape;

		resShape.push_back(a->shape()[0]);

		for (int i = 1; i < b->ndims(); i++) {
			resShape.push_back(b->shape()[i]);
		}

		auto result = new NDArray<T>(resShape);

		for (auto index : ndindex(resShape)) {
			int sum = 0;
			for (int k = 0; k < a->shape()[1]; k++) {
				std::vector<long> i = { index[0], k };
				std::vector<long> j;

				j.push_back(k);

				for (int m = 1; m < index.size(); m++) {
					j.push_back(index[m]);
				}

				T res;
				
				if (a->ndims() == i.size() && b->ndims() == j.size()) {
					res = a->at(i) * b->at(j);
				}
				else if (a->ndims() == i.size()) {
					auto mult = CudheartNew::CPP::Math::BaseMath::multiply(a->byFetchIndices(i), b->byFetchIndices(j));
					res = CudheartNew::CPP::Math::BaseMath::sum(mult);
				}
				sum += res;
			}

			result->at(index) = sum;
		}

		return result;
	}

	template <typename A, typename B, typename T = promote(A, B)>
	NDArray<T>* tensordot(NDArray<A>* a, NDArray<B>* b, Shape a_axes, Shape b_axes) {
		Shape aShape, bShape;

		for (auto axis : a_axes) {
			aShape.push_back(a->shape()[axis]);
		}

		for (auto axis : b_axes) {
			bShape.push_back(b->shape()[axis]);
		}

		if (aShape != bShape) {
			fmt::println("shape-mismatch for sum");
			exit(-1);
		}

		Shape oldA, oldB;

		for (int i = 0; i < a->ndims(); i++) {
			if (std::find(a_axes.begin(), a_axes.end(), i) != a_axes.end()) {
				continue;
			}

			oldA.push_back(i);
		}

		for (int i = 0; i < b->ndims(); i++) {
			if (std::find(b_axes.begin(), b_axes.end(), i) != b_axes.end()) {
				continue;
			}

			oldB.push_back(i);
		}

		auto newAxesA = oldA;
		newAxesA.insert(newAxesA.end(), a_axes.begin(), a_axes.end());

		auto newAxesB = b_axes;
		newAxesB.insert(newAxesB.end(), oldB.begin(), oldB.end());

		auto N2 = 1;

		for (auto dim : aShape) {
			N2 *= dim;
		}

		auto N1 = 1;

		for (auto i : oldA) {
			N1 *= a->shape()[i];
		}

		auto N3 = 1;

		for (auto i : oldB) {
			N3 *= b->shape()[i];
		}

		auto at = a->transpose(newAxesA)->reshape({ N1, N2 }, true);
		auto bt = b->transpose(newAxesB)->reshape({ N2, N3 }, true);
		auto res = dot(at, bt);

		Shape finalShape;

		for (auto i : oldA) {
			finalShape.push_back(a->shape()[i]);
		}

		for (auto i : oldB) {
			finalShape.push_back(b->shape()[i]);
		}

		return res->reshape(finalShape, true);
	}

	template <typename A, typename B, typename T = promote(A, B)>
	NDArray<T>* tensordot(NDArray<A>* a, NDArray<B>* b, int axes = 2) {
		auto na = a->ndims();

		Shape a_axes, b_axes;

		for (int i = na - axes; i < na; i++) {
			a_axes.push_back(i);
		}

		for (int i = 0; i < axes; i++) {
			b_axes.push_back(i);
		}

		return tensordot(a, b, a_axes, b_axes);
	}

	template <typename A, typename B, typename T = promote(A, B)>
	NDArray<T>* inner(NDArray<A>* a, NDArray<B>* b) {
		return tensordot(a, b, { a->ndims() - 1}, { b->ndims() -1});
	}

	template <typename A, typename B, typename T = promote(A, B)>
	NDArray<T>* outer(NDArray<A>* a, NDArray<B>* b) {
		return tensordot(a, b, 0);
	}

	template <typename A, typename B, typename T = promote(A, B)>
	NDArray<T>* matmul(NDArray<A>* a, NDArray<B>* b) {
		return tensordot(a, b, { a->ndims() - 1 }, { std::max(b->ndims() - 2, 0)});
	}

	//template <typename T>
	//T det(Matrix<T>* mat) {
	//	// idfk how numpy implemented their determinant algorithm
	//	// this is a mirror of the algorithm implemented in a plethera of websites
	//	if (mat->getHeight() != mat->getWidth()) {
	//		BaseException("Exception: Matrix has to be square").raise();
	//	}

	//	if (mat->getHeight() == 1) {
	//		return mat->get(0, 0);
	//	}

	//	if (mat->getHeight() == 2) {
	//		return (mat->get(0, 0) * mat->get(1, 1)) - (mat->get(0, 1) * mat->get(1, 0));
	//	}

	//	T value = 0;
	//	int sign = 1;

	//	for (int i = 0; i < mat->getWidth(); i++) {
	//		//mat->print();
	//		//cout << "value: " << mat->get(0, i) << " of index " << i << endl;
	//		Matrix<T>* sub = new Matrix<T>(mat->getHeight() - 1, mat->getWidth() - 1);
	//		int idx = 0;
	//		int jdx = 0;
	//		for (int k = 1; k < mat->getHeight(); k++) {
	//			for (int m = 0; m < mat->getWidth(); m++) {
	//				if (m != i) {
	//					//cout << "m: " << m << " k: " << k << " " << mat->get(k, m) << endl;
	//					//cout << "i: " << idx << " j: " << jdx << endl;
	//					sub->set(jdx, idx, mat->get(k, m));
	//					idx++;
	//					if (idx == mat->getHeight() - 1) {
	//						idx = 0;
	//						jdx++;
	//					}
	//				}
	//			}
	//		}
	//		value += sign * mat->get(0, i) * det(sub);
	//		sign *= -1;
	//	}

	//	return value;
	//}

	//template <typename T>
	//T trace(Matrix<T>* mat, int offset) {
	//	T value = (T)0;

	//	for (int i = 0; i < mat->getHeight() && i + offset < mat->getWidth(); i++) {
	//		value += mat->get(i, i + offset);
	//	}

	//	return value;
	//}

	//template <typename T>
	//T trace(Matrix<T>* mat) {
	//	return trace(mat, 0);
	//}

	//template <typename T>
	//Vector<T>* solve(Matrix<T>* a, Vector<T>* b) {
	//	a->assertMatchShape(b->getShape(), 1);
	//	// relentlessly ripped from https://martin-thoma.com/solving-linear-equations-with-gaussian-elimination/

	//	Matrix<T>* A = new Matrix<T>(a->getHeight(), a->getWidth() + 1);

	//	for (int i = 0; i < a->getHeight(); i++) {
	//		for (int j = 0; j < a->getWidth(); j++) {
	//			A->set(i, j, a->get(i, j));
	//		}
	//	}

	//	for (int i = 0; i < b->size(); i++) {
	//		A->set(i, -1, b->get(i));
	//	}

	//	int n = a->getHeight();

	//	for (int i = 0; i < n; i++) {
	//		// Search for maximum in this column
	//		double maxEl = abs(A->get(i, i));
	//		int maxRow = i;
	//		for (int k = i + 1; k < n; k++) {
	//			if (abs(A->get(k, i)) > maxEl) {
	//				maxEl = abs(A->get(k, i));
	//				maxRow = k;
	//			}
	//		}

	//		// Swap maximum row with current row (column by column)
	//		for (int k = i; k < n + 1; k++) {
	//			double tmp = A->get(maxRow, k);
	//			A->set(maxRow, k, A->get(i, k));
	//			A->set(i, k, tmp);
	//		}

	//		// Make all rows below this one 0 in current column
	//		for (int k = i + 1; k < n; k++) {
	//			double c = -A->get(k, i) / A->get(i, i);
	//			for (int j = i; j < n + 1; j++) {
	//				if (i == j) {
	//					A->set(k, j, 0);
	//				}
	//				else {
	//					A->set(k, j, A->get(k, j) + (c * A->get(i, j)));
	//				}
	//			}
	//		}
	//	}

	//	// Solve equation Ax=b for an upper triangular matrix A
	//	// vector<double> x(n);
	//	Vector<T>* x = new Vector<T>(n);
	//	for (int i = n - 1; i >= 0; i--) {
	//		x->set(i, A->get(i, n) / A->get(i, i));
	//		for (int k = i - 1; k >= 0; k--) {
	//			A->set(k, n, A->get(k, n) - (A->get(k, i) * x->get(i)));
	//		}
	//	}
	//	return x;
	//}

	//template <typename T>
	//T norm(NDArray<T>* x) {
	//	T sum = (T)0;

	//	for (int i = 0; i < x->size(); i++) {
	//		sum += std::pow(x->get(i), 2);
	//	}

	//	return std::sqrt(sum);
	//}

	//namespace {
	//	template <typename T>
	//	Vector<T>* full_norm(NDArray<T>* x, int n) {
	//		return full(1, n, norm<T>(x))->flatten();
	//	}

	//	template <typename T>
	//	std::pair<T, Vector<T>*> power_method(Matrix<T>* A, double epsilon, int maxIter) {
	//		int n = A->getHeight();
	//		std::pair<T, Vector<T>*> out;
	//		Vector<T>* x = Cudheart::CPP::Random::random<T>(n);
	//		Vector<T>* full = full_norm(x, n);
	//		x = (Vector<T>*)BaseMath::divide(x, full);
	//		T newLambda = 0;

	//		for (int i = 0; i < maxIter; i++) {
	//			Vector<T>* newX = dot(A, x);
	//			newLambda = dot(newX, x);
	//			newX = (Vector<T>*)BaseMath::divide(newX, full_norm(newX, n));
	//			if (norm(BaseMath::subtract(newX, x)) < epsilon) {
	//				x = newX;
	//				break;
	//			}
	//			x = newX;
	//		}

	//		out.first = newLambda;
	//		out.second = x;

	//		return out;
	//	}
	//}

	//template <typename T>
	//std::pair<Vector<T>*, Vector<T>**> eig(Matrix<T>* A) {
	//	int n = A->getHeight();
	//	Vector<T>* eigenvalues = new Vector<T>(n);
	//	Vector<T>** eigenvectors = new Vector<T>*[n];
	//	std::pair<Vector<T>*, Vector<T>**> out;

	//	for (int i = 0; i < n; i++) {
	//		std::pair<T, Vector<T>*> pair = power_method(A, 1e-8, 1000);
	//		Vector<T>* se = pair.second;
	//		eigenvalues->set(i, pair.first);
	//		eigenvectors[i] = pair.second;
	//		Matrix<T>* o = outer(pair.second, pair.second);
	//		Matrix<T>* lm = full(o->getHeight(), o->getWidth(), pair.first);
	//		Matrix<T>* mult = (Matrix<T>*)BaseMath::multiply(lm, o);
	//		A = (Matrix<T>*)BaseMath::subtract(A, mult);
	//	}

	//	out.first = eigenvalues;
	//	out.second = eigenvectors;

	//	return out;
	//}

	//template <typename T>
	//Vector<T>* eigvals(Matrix<T>* A) {
	//	std::pair<Vector<T>*, Vector<T>**> pair = eig(A);
	//	return pair.first;
	//}

	//template <typename T>
	//Vector<T>* roots(NDArray<T>* p) {
	//	int N = p->size();

	//	int start = 0;
	//	int end = p->size();
	//	int trailing = 0;

	//	for (int i = 0; i < p->size(); i++) {
	//		if (p->get(i) != 0) {
	//			break;
	//		}
	//		start++;
	//	}

	//	for (int i = p->size() - 1; i >= 0; i--) {
	//		if (p->get(i) != 0) {
	//			break;
	//		}
	//		end--;
	//		trailing++;
	//	}

	//	Vector<T>* newP = new Vector<T>(end - start);
	//	int index = 0;

	//	for (int i = start; i < end; i++) {
	//		newP->set(index++, p->get(i));
	//	}

	//	p = newP;

	//	Vector<T>* a = ones<T>(N - 2);

	//	Matrix<T>* A = diagflat<T>(a, -1);

	//	for (int i = 0; i < A->getWidth(); i++) {
	//		A->set(0, i, -(p->get(i + 1) / p->get(0)));
	//	}

	//	return eigvals(A);
	//}

	//template <typename T>
	//Matrix<T>* inv(Matrix<T>* A) {
	//	Matrix<T>* mat = (Matrix<T>*)A->copy();
	//	if (mat->getHeight() != mat->getWidth()) {
	//		BaseException("Exception: Matrix has to be square").raise();
	//	}
	//	int n = mat->getHeight();

	//	Matrix<T>* result = zeros<T>(n, n);

	//	// Create the identity matrix
	//	for (int i = 0; i < n; i++) {
	//		result->set(i, i, 1);
	//	}

	//	// Gauss-Jordan elimination to transform A into the identity matrix
	//	for (int i = 0; i < n; i++) {
	//		// Find the pivot row
	//		int pivot = i;
	//		for (int j = i + 1; j < n; j++) {
	//			if (fabs(mat->get(j, i)) > fabs(mat->get(pivot, i))) {
	//				pivot = j;
	//			}
	//		}

	//		// Swap the pivot row with the current row
	//		Vector<T>* temp = new Vector<T>(n);

	//		for (int j = 0; j < n; j++) {
	//			temp->set(j, mat->get(pivot, j));
	//		}
	//		for (int j = 0; j < n; j++) {
	//			mat->set(pivot, j, mat->get(i, j));
	//		}
	//		for (int j = 0; j < n; j++) {
	//			mat->set(i, j, temp->get(j));
	//		}

	//		for (int j = 0; j < n; j++) {
	//			temp->set(j, result->get(pivot, j));
	//		}
	//		for (int j = 0; j < n; j++) {
	//			result->set(pivot, j, result->get(i, j));
	//		}
	//		for (int j = 0; j < n; j++) {
	//			result->set(i, j, temp->get(j));
	//		}

	//		// Normalize the pivot row
	//		T pivotValue = mat->get(i, i);
	//		for (int j = 0; j < n; j++) {
	//			mat->set(i, j, mat->get(i, j) / pivotValue);
	//			result->set(i, j, result->get(i, j) / pivotValue);
	//		}

	//		// Eliminate the current column
	//		for (int j = 0; j < n; j++) {
	//			if (j == i) continue;

	//			T ratio = mat->get(j, i);
	//			for (int k = 0; k < n; k++) {
	//				mat->set(j, k, mat->get(j, k) - (ratio * mat->get(i, k)));
	//				result->set(j, k, result->get(j, k) - (ratio * result->get(i, k)));
	//			}
	//		}
	//	}

	//	delete mat;

	//	return result;
	//}

	//template <typename T>
	//Vector<T>* convolve(Vector<T>* a, Vector<T>* b) {
	//	Vector<T>* out = Cudheart::VectorOps::zeros<T>(a->size() + b->size() - 1);

	//	for (int i = 0; i < a->size(); i++) {
	//		for (int j = 0; j < b->size(); j++) {
	//			T last = out->get(i + j);
	//			out->set(i + j, last + (a->get(i) * b->get(j)));
	//		}
	//	}

	//	return out;
	//}

	//template <typename T>
	//NDArray<T>* clip(NDArray<T>* arr, T min, T max) {
	//	NDArray<T>* out = arr->copy();

	//	for (int i = 0; i < out->size(); i++) {
	//		if (out->get(i) < min) {
	//			out->set(i, min);
	//		}
	//		else if (out->get(i) > max) {
	//			out->set(i, max);
	//		}
	//	}

	//	return out;
	//}

	//template <typename T>
	//NDArray<T>* clip(NDArray<T>* arr, T max) {
	//	NDArray<T>* out = arr->copy();

	//	for (int i = 0; i < out->size(); i++) {
	//		if (out->get(i) > max) {
	//			out->set(i, max);
	//		}
	//	}

	//	return out;
	//}
}