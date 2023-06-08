#pragma once

#include "NDArray.cuh"

#include <vector>

#define FMT_HEADER_ONLY
#include "fmt/core.h"

#include "../Internal/Promotion.cuh"

namespace CudheartNew::ArrayOps {
	template <typename T>
	NDArray<T>* asArray(std::vector<T> data) {
		auto out = new NDArray<T>({ (int)data.size() });

		for (int i = 0; i < data.size(); i++) {
			out->at(i) = data.at(i);
		}

		return out;
	}

	template <typename T>
	NDArray<T>* empty(std::vector<int> shape) {
		return new NDArray<T>(shape);
	}

	template <typename T>
	NDArray<T>* emptyLike(NDArray<T>* arr) {
		return new NDArray<T>(arr->shape());
	}

	template <typename A, typename B>
	NDArray<B>* emptyLike(NDArray<A>* arr) {
		return new NDArray<B>(arr->shape());
	}

	template <typename T>
	NDArray<T>* arange(T start, T end, T jump) {
		if (signbit((double)(end - start)) != signbit((double)jump)) {
			fmt::println("cannot define range from {} to {}", start, end);
			exit(1);
		}

		// thanks c++
		float diff = (float)end - (float)start;
		float steps = diff / (float)jump;
		auto len = (int)ceil(steps);

		NDArray<T>* out = new NDArray<T>({ len });

		for (int i = 0; i < len; start += jump, i++) {
			out->at(i) = start;
		}

		return out;
	}

	template <typename T>
	NDArray<T>* arange(T start, T end) {
		return arange(start, end, (T)1);
	}

	template <typename T>
	NDArray<T>* arange(T end) {
		return arange((T)0, end, (T)1);
	}

	template <typename T>
	NDArray<T>* full(std::vector<int> shape, T value) {
		NDArray<T>* out = new NDArray<T>(shape);

		for (int i = 0; i < out->size(); i++) {
			out->at(i) = value;
		}

		return out;
	}

	template <typename T>
	NDArray<T>* fullLike(NDArray<T>* arr, T value) {
		return fullLike(arr->shape(), value);
	}

	template <typename T>
	NDArray<T>* linspace(T start, T stop, int num, bool endpoint) {
		double diff = (double)stop - (double)start;

		NDArray<T>* out = new NDArray<T>({ num });

		if (endpoint) {
			num--;
			out->at(-1) = stop;
		}

		T jump = (T)(diff / num);

		for (int i = 0; i < num; i++) {
			out->at(i) = (T)(start + jump * i);
		}

		return out;
	}

	template <typename T>
	NDArray<T>* linspace(T start, T stop, int num) {
		return linspace<T>(start, stop, num, true);
	}

	template <typename T>
	NDArray<T>* linspace(T start, T stop) {
		return linspace<T>(start, stop, 50, true);
	}

	template <typename T>
	NDArray<T>* ones(std::vector<int> shape) {
		return full(shape, 1);
	}

	template <typename T>
	NDArray<T>* onesLike(NDArray<T>* arr) {
		return ones(arr->shape());
	}

	template <typename T>
	NDArray<T>* zeros(std::vector<int> shape) {
		return full(shape, 0);
	}

	template <typename T>
	NDArray<T>* zerosLike(NDArray<T>* arr) {
		return zeros(arr->shape());
	}

	template <typename T>
	NDArray<T>* logspace(T start, T stop, int num, bool endpoint, T base) {
		double diff = (double)stop - (double)start;

		NDArray<T>* out = new NDArray<T>({ num });

		if (endpoint) {
			num--;
			out->at(-1) = pow(base, stop);
		}

		T jump = (T)(diff / num);

		for (int i = 0; i < num; i++) {
			out->at(i) = pow(base, (T)(start + jump * i));
		}

		return out;
	}

	template <typename T>
	NDArray<T>* logspace(T start, T stop, int num, bool endpoint) {
		return logspace(start, stop, num, endpoint, 10.0);
	}

	template <typename T>
	NDArray<T>* logspace(T start, T stop, int num, double base) {
		return logspace(start, stop, num, true, base);
	}

	template <typename T>
	NDArray<T>* logspace(T start, T stop, int num) {
		return logspace(start, stop, num, true, 10.0);
	}

	template <typename T>
	NDArray<T>* logspace(T start, T stop) {
		return logspace(start, stop, 50, true, 10.0);
	}

	template <typename T>
	NDArray<T>* geomspace(T start, T stop, int num, bool endpoint) {
		return logspace((T)log10(start), (T)log10(stop), num, endpoint, (T)10.0);
	}

	template <typename T>
	NDArray<T>* geomspace(T start, T stop, int num) {
		return geomspace<T>(start, stop, num, true);
	}

	template <typename T>
	NDArray<T>* geomspace(T start, T stop) {
		return geomspace<T>(start, stop, (T)50, true);
	}

	template <typename T>
	NDArray<T>* eye(int N, int M, int k) {
		auto out = new NDArray<T>({ N, M });

		for (int i = 0, j = k; i < N && j < M; i++, j++) {
			j %= M;
			out->at({ i, j }) = (T)1;
		}

		return out;
	}

	template <typename T>
	NDArray<T>* eye(int N, int k) {
		return eye<T>(N, N, k);
	}

	template <typename T>
	NDArray<T>* eye(int N) {
		return eye<T>(N, N, 0);
	}

	template <typename T, typename U, typename K>
	std::vector<NDArray<T>*> meshgrid(NDArray<U>* a, NDArray<K>* b) {
		a = a->flatten();
		b = b->flatten();
		std::vector<NDArray<T>*> out;
		NDArray<T>* first = new NDArray<T>({ b->size(), a->size() });
		NDArray<T>* second = new NDArray<T>({ b->size(), a->size() });

		out[0] = first;
		out[1] = second;

		for (int i = 0; i < b->size(); i++) {
			for (int j = 0; j < a->size(); j++) {
				first->at({ i, j }) = a->at(j);
			}
		}

		for (int i = 0; i < b->size(); i++) {
			for (int j = 0; j < a->size(); j++) {
				second->at({ i, j }) = a->at(i);
			}
		}

		return out;
	}

	template <typename T>
	NDArray<T>* diag(NDArray<T>* v, int k = 0) {
		if (v->ndims() > 2) {
			fmt::println("ndarray of {} dims is not allowed in diag", v->ndims());
			exit(-1);
		}
		NDArray<T>* out;
		if (v->ndims() == 2) {
			auto size = std::min(v->shape()[0], v->shape()[1]);
			out = new NDArray<T>({ size });

			for (int i = 0; i < size; i++) {
				out->at(i) = v->at(i + k);
			}
		}
		else {
			out = new NDArray<T>({ v->size(), v->size() });

			for (int i = 0; i < v->size(); i++) {
				out->at({ i, i + k }) = v->at(i);
			}
		}

		return out;
	}

	template <typename T>
	NDArray<T>* diagflat(NDArray<T>* v, int k = 0) {
		return diag(v->flatten());
	}

	template <typename T>
	NDArray<T>* tri(int N, int M, int k) {
		NDArray<T>* out = zeros<T>({ N, M });

		for (int i = 0; i < N; i++) {
			for (int j = 0; j <= k + i && j < M; j++) {
				out->at({ i, j }) = 1;
			}
		}

		return out;
	}

	template <typename T>
	NDArray<T>* tri(int N, int M) {
		return tri<T>(N, M, 0);
	}

	template <typename T>
	NDArray<T>* tri(int N) {
		return tri<T>(N, N, 0);
	}

	template <typename T>
	NDArray<T>* tril(NDArray<T>* mat, int k = 0) {
		if (mat->ndims() != 2) {
			fmt::println("non matrix arrays are not supported in tril");
			exit(-1);
		}
		NDArray<T>* out = mat->copy();

		for (int i = 0; i < mat->shape()[0]; i++) {
			for (int j = k + i + 1; j < mat->shape()[1]; j++) {
				out->at({ i, j }) = 0;
			}
		}

		return out;
	}

	template <typename T>
	NDArray<T>* vander(NDArray<T>* vec, int N, bool increasing) {
		if (vec->ndims() != 1) {
			fmt::println("non vector arrays are not supported in vander");
			exit(-1);
		}

		NDArray<T>* out = new NDArray<T>({ vec->size(), N });

		if (increasing) {
			for (int i = 0; i < out->shape()[0]; i++) {
				for (int j = 0; j < out->shape()[1]; j++) {
					out->at({ i, j }) = std::pow(vec->at(i), j);
				}
			}
		}
		else {
			for (int i = 0; i < out->shape()[0]; i++) {
				for (int j = out->shape()[1] - 1; j >= 0; j--) {
					out->at({ i, j }) = std::pow(vec->at(i), j);
				}
			}
		}

		return out;
	}

	template <typename T>
	NDArray<T>* vander(NDArray<T>* vec, int N) {
		return vander(vec, N, false);
	}

	template <typename T>
	NDArray<T>* vander(NDArray<T>* vec, bool increasing) {
		return vander(vec, vec->size(), increasing);
	}

	template <typename T>
	NDArray<T>* vander(NDArray<T>* vec) {
		return vander(vec, vec->size(), false);
	}

	std::vector<std::vector<int>> ndindex(std::vector<int> shape) {
		std::vector<std::vector<int>> out;

		std::vector<int> s(shape.size(), 0);

		int size = 1;

		for (auto val : shape) {
			size *= val;
		}

		out.reserve(size);

		for (int i = 0; i < size; i++) {
			out.push_back(s);

			for (int j = s.size() - 1; j >= 0; j--) {
				s.at(j)++;

				if (s.at(j) == shape.at(j)) {
					s.at(j) = 0;
				}
				else {
					break;
				}
			}
		}

		return out;
	}

	template <typename T>
	NDArray<T>* take(NDArray<T>* a, NDArray<int>* indices, int axis = -1) {
		if (axis == -1) {
			auto out = new NDArray<T>(indices->shape());
			for (int i = 0; i < indices->size(); i++) {
				out->at(i) = a->at(indices->at(i));
			}

			return out;
		}

		std::vector<int> Ni;

		for (int i = 0; i < axis; i++) {
			Ni.push_back(a->shape().at(i));
		}

		std::vector<int> Nk;

		for (int i = axis + 1; i < a->ndims(); i++) {
			Nk.push_back(a->shape().at(i));
		}

		auto Nj = indices->shape();

		std::vector<int> outShape;

		outShape.insert(outShape.begin(), Ni.begin(), Ni.end());
		outShape.insert(outShape.end(), Nj.begin(), Nj.end());
		outShape.insert(outShape.end(), Nk.begin(), Nk.end());

		auto out = new NDArray<T>(outShape);

		for (auto& ii : ndindex(Ni)) {
			for (auto& jj : ndindex(Nj)) {
				for (auto& kk : ndindex(Nk)) {
					std::vector<int> outIndex;

					outIndex.insert(outIndex.begin(), ii.begin(), ii.end());
					outIndex.insert(outIndex.end(), jj.begin(), jj.end());
					outIndex.insert(outIndex.end(), kk.begin(), kk.end());

					std::vector<int> inIndex;

					inIndex.insert(inIndex.begin(), ii.begin(), ii.end());
					inIndex.push_back(indices->at(jj));
					inIndex.insert(inIndex.end(), kk.begin(), kk.end());

					// fmt::println("{}, {}", fmt::join(outIndex, ","), fmt::join(inIndex, ","));

					out->at(outIndex) = a->at(inIndex);
				}
			}
		}

		return out;
	}

	template <typename T>
	NDArray<T>* take(NDArray<T>* a, int index, int axis = -1) {
		auto indices = new NDArray<int>({ 1 });
		indices->at(0) = index;

		return take(a, indices, axis);
	}

	template <typename T>
	NDArray<T>* concatenate(std::vector<NDArray<T>*>& seq, int axis = 0) {
		auto shape = seq.at(0)->shape();

		shape.at(axis) = 0;

		for (auto arr : seq) {
			auto s = arr->shape();
			shape.at(axis) += s.at(axis);
		}

		auto out = new NDArray<T>(shape);

		int index = 0;
		for (NDArray<T>* arr : seq) {
			auto size = arr->shape().at(axis);

			for (int i = 0; i < size; i++) {
				auto indices = ndindex(shape);
				auto taken = take<T>(arr, i, axis);

				int j = 0;
				for (auto idx : indices) {
					if (idx[axis] != index + i) {
						continue;
					}

					out->at(idx) = taken->at(j++);
				}
			}

			index += size;
		}

		return out;
	}

	template <typename T>
	NDArray<T>* append(NDArray<T>* a, T b) {
		auto out = new NDArray<T>({ a->size() + 1 });

		for (int i = 0; i < a->size(); i++) {
			out->at(i) = a->at(i);
		}

		out->at(-1) = b;

		return out;
	}

	template <typename T>
	NDArray<T>* append(NDArray<T>* a, NDArray<T>* b, int axis = -1) {
		if (axis == -1) {
			a = a->flatten();
			b = b->flatten();
			axis = 0;
		}

		std::vector<NDArray<int>*> vec = { a, b };

		return concatenate<T>(vec, axis);
	}

	template <typename T>
	NDArray<T>* trimZeros(NDArray<T>* filt, std::string trim = "fb") {
		int start = 0;
		int end = filt->size();

		if (trim.find("f") != std::string::npos) {
			for (int i = 0; i < filt->size(); i++) {
				if (filt->at(i) != 0) {
					start = i;
					break;
				}
			}
		}

		if (trim.find("b") != std::string::npos) {
			for (int i = filt->size() - 1; i > start; i--) {
				if (filt->at(i) != 0) {
					end = i + 1;
					break;
				}
			}
		}

		int size = end - start;

		NDArray<T>* out = new NDArray<T>({ size });

		for (int i = 0; i < size; i++) {
			out->at(i) = filt->at(i + start);
		}

		return out;
	}

	namespace {
		// :,(
		template <typename T>
		int partition(NDArray<T>* arr, int low, int high) {
			T pivot = arr->at(high);
			int i = (low - 1);

			for (int j = low; j <= high - 1; j++)
			{
				// If current element is smaller than the pivot
				if (arr->at(j) < pivot)
				{
					i++; // increment index of smaller element
					T temp = arr->at(i);
					arr->at(i) = arr->at(j);
					arr->at(j) = temp;
				}
			}
			T temp = arr->at(i + 1);
			arr->at(i + 1) = arr->at(high);
			arr->at(high) = temp;
			return (i + 1);
		}

		template <typename T>
		void quicksort(NDArray<T>* arr, int low, int high) {
			if (low < high) {
				int pi = partition(arr, low, high);

				quicksort(arr, low, pi - 1);
				quicksort(arr, pi + 1, high);
			}
		}
	}

	template <typename T>
	std::vector<NDArray<T>*> unique(NDArray<T>* ar, bool returnIndex, bool returnInverse, bool returnCounts) {
		std::vector<NDArray<T>*> vectors;

		int uniques = 0;

		for (int i = 0; i < ar->size(); i++) {
			bool isUnique = true;
			for (int j = 0; j < i; j++) {
				if (ar->at(i) == ar->at(j)) {
					isUnique = false;
				}
			}

			if (isUnique) {
				uniques++;
			}
		}

		auto uniqueArr = new NDArray<T>({ uniques });
		auto indexArr = new NDArray<T>({ uniques });
		auto inverseArr = new NDArray<T>({ ar->shape() });
		auto countsArr = zeros<T>({ uniques });

		int index = 0;

		for (int i = 0; i < ar->size(); i++) {
			bool unique = true;
			for (int j = 0; j < index; j++) {
				if (uniqueArr->at(j) == ar->at(i)) {
					unique = false;
					break;
				}
			}

			if (unique) {
				uniqueArr->at(index++) = ar->at(i);

				if (index == uniques) {
					break;
				}
			}
		}

		quicksort(uniqueArr, 0, uniques - 1);

		if (returnIndex) {
			for (int i = 0; i < uniqueArr->size(); i++) {
				for (int j = 0; j < ar->size(); j++) {
					if (ar->at(j) == uniqueArr->at(i)) {
						indexArr->at(i) = j;
						break;
					}
				}
			}
		}

		if (returnInverse) {
			for (int i = 0; i < ar->size(); i++) {
				for (int j = 0; j < uniqueArr->size(); j++) {
					if (ar->at(i) == uniqueArr->at(j)) {
						inverseArr->at(i) == j;
					}
				}
			}
		}

		if (returnCounts) {
			for (int i = 0; i < ar->size(); i++) {
				for (int j = 0; j < uniqueArr->size(); j++) {
					if (ar->at(i) == uniqueArr->at(j)) {
						countsArr->at(j) = countsArr->at(j) + 1;
					}
				}
			}
		}

		vectors.push_back(uniqueArr);
		returnIndex ? vectors.push_back(indexArr) : nullptr;
		returnInverse ? vectors.push_back(inverseArr) : nullptr;
		returnCounts ? vectors.push_back(countsArr) : nullptr;

		return vectors;
	}

	template <typename T>
	std::vector<NDArray<T>*> unique(NDArray<T>* ar, bool returnIndex, bool returnInverse) {
		return unique<T>(ar, returnIndex, returnInverse, false);
	}

	template <typename T>
	std::vector<NDArray<T>*> unique(NDArray<T>* ar, bool returnIndex) {
		return unique<T>(ar, returnIndex, false, false);
	}

	template <typename T>
	std::vector<NDArray<T>*> unique(NDArray<T>* ar) {
		return unique<T>(ar, false, false, false);
	}
}