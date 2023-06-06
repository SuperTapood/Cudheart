#pragma once

#include "../Arrays/NDArray.cuh"
#include <time.h>
#include <stdlib.h>
// note! some algorithms have a parallel implementation. take a look at that pls.

namespace CudheartNew::Sorting {
	enum class Kind {
		Quicksort, Mergesort, Heapsort
	};

	namespace {
		template <typename T>
		int _partition(NDArray<T>* arr, int low, int high) {
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
			return i + 1;
		}

		template <typename T>
		void _quicksort(NDArray<T>* arr, int low, int high) {
			if (low < high) {
				int pi = _partition(arr, low, high);

				_quicksort(arr, low, pi - 1);
				_quicksort(arr, pi + 1, high);
			}
		}

		template <typename T>
		void _merge(NDArray<T>* arr, int left, int mid, int right) {
			auto const subArrayOne = mid - left + 1;
			auto const subArrayTwo = right - mid;

			// Create temp arrays
			auto* leftArray = new T[subArrayOne],
				* rightArray = new T[subArrayTwo];

			// Copy data to temp arrays leftArray[] and rightArray[]
			for (auto i = 0; i < subArrayOne; i++)
				leftArray[i] = arr->at(left + i);
			for (auto j = 0; j < subArrayTwo; j++)
				rightArray[j] = arr->at(mid + 1 + j);

			auto indexOfSubArrayOne = 0, // Initial index of first sub-array
				indexOfSubArrayTwo = 0; // Initial index of second sub-array
			int indexOfMergedArray = left; // Initial index of merged array

			// Merge the temp arrays back into array[left..right]
			while (indexOfSubArrayOne < subArrayOne && indexOfSubArrayTwo < subArrayTwo) {
				if (leftArray[indexOfSubArrayOne] <= rightArray[indexOfSubArrayTwo]) {
					arr->at(indexOfMergedArray) = leftArray[indexOfSubArrayOne];
					indexOfSubArrayOne++;
				}
				else {
					arr->at(indexOfMergedArray) = rightArray[indexOfSubArrayTwo];
					indexOfSubArrayTwo++;
				}
				indexOfMergedArray++;
			}
			// Copy the remaining elements of
			// left[], if there are any
			while (indexOfSubArrayOne < subArrayOne) {
				arr->at(indexOfMergedArray) = leftArray[indexOfSubArrayOne];
				indexOfSubArrayOne++;
				indexOfMergedArray++;
			}
			// Copy the remaining elements of
			// right[], if there are any
			while (indexOfSubArrayTwo < subArrayTwo) {
				arr->at(indexOfMergedArray) = rightArray[indexOfSubArrayTwo];
				indexOfSubArrayTwo++;
				indexOfMergedArray++;
			}
		}

		template <typename T>
		void _mergesort(NDArray<T>* arr, int begin, int end) {
			if (begin >= end) {
				return;
			}

			auto mid = begin + (end - begin) / 2;
			_mergesort(arr, begin, mid);
			_mergesort(arr, mid + 1, end);
			_merge(arr, begin, mid, end);
		}

		template <typename T>
		void _heapify(NDArray<T>* arr, int n, int i) {
			// Initialize largest as root
			int largest = i;

			// left = 2*i + 1
			int l = 2 * i + 1;

			// right = 2*i + 2
			int r = 2 * i + 2;

			// If left child is larger than root
			if (l < n && arr->at(l) > arr->at(largest))
				largest = l;

			// If right child is larger than largest
			// so far
			if (r < n && arr->at(r) > arr->at(largest))
				largest = r;

			// If largest is not root
			if (largest != i) {
				T temp = arr->at(i);
				arr->at(i) = arr->at(largest);
				arr->at(largest) = temp;

				// Recursively heapify the affected
				// sub-tree
				_heapify(arr, n, largest);
			}
		}

		template <typename T>
		void _heapsort(NDArray<T>* arr, int n) {
			// Build heap (rearrange array)
			for (int i = n / 2 - 1; i >= 0; i--)
				_heapify(arr, n, i);

			// One by one extract an element
			// from heap
			for (int i = n - 1; i > 0; i--) {
				// Move current root to end
				T temp = arr->at(0);
				arr->at(0) = arr->at(i);
				arr->at(i) = temp;

				// call max heapify on the reduced heap
				_heapify(arr, i, 0);
			}
		}
	}

	template <typename T>
	NDArray<T>* quicksort(NDArray<T>* a) {
		NDArray<T>* arr = a->copy();
		_quicksort(arr, 0, a->size() - 1);
		return arr;
	}

	template <typename T>
	NDArray<T>* mergesort(NDArray<T>* a) {
		NDArray<T>* arr = a->copy();
		_mergesort(arr, 0, a->size() - 1);
		return arr;
	}

	template <typename T>
	NDArray<T>* heapsort(NDArray<T>* a) {
		NDArray<T>* arr = a->copy();
		_heapsort(arr, a->size());
		return arr;
	}

	template <typename T>
	NDArray<T>* sort(NDArray<T>* a, Kind kind = Kind::Quicksort) {
		switch (kind) {
		case Kind::Quicksort:
			return quicksort(a);
		case Kind::Mergesort:
			return mergesort(a);
		case Kind::Heapsort:
			return heapsort(a);
		default:
			return quicksort(a);
		}

		// should never be returned :)
		return nullptr;
	}

	template <typename T>
	NDArray<T>* argsort(NDArray<T>* arr, Kind kind = Kind::Quicksort) {
		NDArray<T>* sorted = sort(arr, kind);

		NDArray<T>* indices = new NDArray<T>(arr->shape());
		T temp = sorted->at(0) - 1;

		for (int i = 0; i < indices->size(); i++) {
			for (int j = 0; j < sorted->size(); j++) {
				if (arr->at(i) == sorted->at(j)) {
					indices->at(j) = i;
					sorted->at(j) = temp;
					break;
				}
			}
		}

		delete sorted;

		return indices;
	}

	template <typename T>
	NDArray<T>* partition(NDArray<T>* a, int kth) {
		NDArray<T>* out = new NDArray<T>(a->shape());

		auto sorted = sort(a);
		T element = sorted->at(kth);
		int right = 0;
		int left = kth;

		out->set(kth, element);

		for (int i = 0; i < out->size(); i++) {
			if (sorted->at(i) <= element) {
				out->at(right++) = sorted->at(i);
			}
			else {
				out->at(left++) = sorted->at(i);
			}
		}

		return out;
	}

	template <typename T>
	NDArray<T>* argpartition(NDArray<T>* a, int kth) {
		NDArray<T>* indices = ArrayOps::arange(a->shape());

		T element = a->at(kth);
		int start = 0;
		int end = a->size() - 1;
		for (int i = 0; i < a->size(); i++) {
			if (i == kth) {
				continue;
			}
			T v = a->at(i);

			if (v < element) {
				indices->at(start) = i;
				start++;
			}
			else {
				indices->at(end) = i;
				end--;
			}
		}

		// start + 1 == end
		indices->at(start + 1) = kth;

		return indices;
	}
}