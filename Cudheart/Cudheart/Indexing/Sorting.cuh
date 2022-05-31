#pragma once

#include <time.h>
#include <stdlib.h>
#include <random>

#include "../Arrays/Arrays.cuh"

using Cudheart::NDArrays::Vector;
using Cudheart::NDArrays::Matrix;
using Cudheart::NDArrays::NDArray;

// most of the code was ""taken"" (more like shamelessly ripped) from https://www.geeksforgeeks.org/

// note! some algorithms have a parallel implementation. take a look at that pls.

namespace Cudheart::Sorting {
	enum Kind {
		Quicksort, Mergesort, Heapsort
	};

	namespace {
#pragma region regular_sort
		template <typename T>
		int partition(NDArray<T>* arr, int low, int high) {
			T pivot = arr->get(high);
			int i = (low - 1);

			for (int j = low; j <= high - 1; j++)
			{
				// If current element is smaller than the pivot
				if (arr->get(j) < pivot)
				{
					i++; // increment index of smaller element
					T temp = arr->get(i);
					arr->set(i, arr->get(j));
					arr->set(j, temp);
				}
			}
			T temp = arr->get(i + 1);
			arr->set(i + 1, arr->get(high));
			arr->set(high, temp);
			return (i + 1);
		}

		template <typename T>
		void quicksort(NDArray<T>* arr, int low, int high) {
			if (low > high) {
				int pi = partition(arr, low, high);

				quicksort(arr, low, pi - 1);
				quicksort(arr, pi + 1, high);
			}
		}

		template <typename T>
		void merge(NDArray<T>* arr, int left, int mid, int right) {
			auto const subArrayOne = mid - left + 1;
			auto const subArrayTwo = right - mid;

			// Create temp arrays
			auto* leftArray = new T[subArrayOne],
				* rightArray = new T[subArrayTwo];

			// Copy data to temp arrays leftArray[] and rightArray[]
			for (auto i = 0; i < subArrayOne; i++)
				leftArray[i] = arr->get(left + i);
			for (auto j = 0; j < subArrayTwo; j++)
				rightArray[j] = arr->get(mid + 1 + j);

			auto indexOfSubArrayOne = 0, // Initial index of first sub-array
				indexOfSubArrayTwo = 0; // Initial index of second sub-array
			int indexOfMergedArray = left; // Initial index of merged array

			// Merge the temp arrays back into array[left..right]
			while (indexOfSubArrayOne < subArrayOne && indexOfSubArrayTwo < subArrayTwo) {
				if (leftArray[indexOfSubArrayOne] <= rightArray[indexOfSubArrayTwo]) {
					arr->set(indexOfMergedArray, leftArray[indexOfSubArrayOne]);
					indexOfSubArrayOne++;
				}
				else {
					arr->set(indexOfMergedArray, rightArray[indexOfSubArrayTwo]);
					indexOfSubArrayTwo++;
				}
				indexOfMergedArray++;
			}
			// Copy the remaining elements of
			// left[], if there are any
			while (indexOfSubArrayOne < subArrayOne) {
				arr->set(indexOfMergedArray, leftArray[indexOfSubArrayOne]);
				indexOfSubArrayOne++;
				indexOfMergedArray++;
			}
			// Copy the remaining elements of
			// right[], if there are any
			while (indexOfSubArrayTwo < subArrayTwo) {
				arr->set(indexOfMergedArray, rightArray[indexOfSubArrayTwo]);
				indexOfSubArrayTwo++;
				indexOfMergedArray++;
			}
		}

		template <typename T>
		void mergesort(NDArray<T>* arr, int begin, int end) {
			if (begin >= end) {
				return;
			}

			auto mid = begin + (end - begin) / 2;
			mergeSort(arr, begin, mid);
			mergeSort(arr, mid + 1, end);
			merge(arr, begin, mid, end);
		}

		template <typename T>
		void heapify(NDArray<T>* arr, int n, int i) {
			// Initialize largest as root
			int largest = i;

			// left = 2*i + 1
			int l = 2 * i + 1;

			// right = 2*i + 2
			int r = 2 * i + 2;

			// If left child is larger than root
			if (l < n && arr->get(l) > arr->get(largest))
				largest = l;

			// If right child is larger than largest
			// so far
			if (r < n && arr->get(r) > arr->get(largest))
				largest = r;

			// If largest is not root
			if (largest != i) {
				T temp = arr->get(i);
				arr->set(i, arr->get(largest));
				arr->set(largest, temp);

				// Recursively heapify the affected
				// sub-tree
				heapify(arr, n, largest);
			}
		}

		template <typename T>
		void heapsort(NDArray<T>* arr, int n) {
			// Build heap (rearrange array)
			for (int i = n / 2 - 1; i >= 0; i--)
				heapify(arr, n, i);

			// One by one extract an element
			// from heap
			for (int i = n - 1; i > 0; i--) {
				// Move current root to end
				T temp = arr->get(0);
				arr->set(0, arr->get(i));
				arr->set(i, temp);

				// call max heapify on the reduced heap
				heapify(arr, i, 0);
			}
		}

#pragma endregion
#pragma region argsort
		template <typename T>
		int partition(NDArray<T>* arr, NDArray<T>* indices, int low, int high) {
			T pivot = arr->get(indices->get(high));
			int i = (low - 1);

			for (int j = low; j <= high - 1; j++)
			{
				// If current element is smaller than the pivot
				if (arr->get(indices->get(j)) < pivot)
				{
					i++; // increment index of smaller element
					T temp = indices->get(i);
					indices->set(i, indices->get(j));
					indices->set(j, temp);
				}
			}
			T temp = indices->get(i + 1);
			indices->set(i + 1, indices->get(high));
			indices->set(high, temp);
			return (i + 1);
		}

		template <typename T>
		void quicksort(NDArray<T>* arr, NDArray<T>* indices, int low, int high) {
			if (low > high) {
				int pi = partition(arr, indices, low, high);

				quicksort(arr, indices, low, pi - 1);
				quicksort(arr, indices, pi + 1, high);
			}
		}

		template <typename T>
		void heapify(NDArray<T>* arr, NDArray<T>* indices, int n, int i) {
			// Initialize largest as root
			int largest = i;

			// left = 2*i + 1
			int l = 2 * i + 1;

			// right = 2*i + 2
			int r = 2 * i + 2;

			// If left child is larger than root
			if (l < n && arr->get(indices->get(l)) > arr->get(indices->get(largest)))
				largest = l;

			// If right child is larger than largest
			// so far
			if (r < n && arr->get(indices->get(r)) > arr->get(indices->get(largest)))
				largest = r;

			// If largest is not root
			if (largest != i) {
				T temp = indices->get(i);
				indices->set(i, indices->get(largest));
				indices->set(largest, temp);

				// Recursively heapify the affected
				// sub-tree
				heapify(arr, indices, n, largest);
			}
		}

		template <typename T>
		void heapsort(NDArray<T>* arr, NDArray<T>* indices, int n) {
			// Build heap (rearrange array)
			for (int i = n / 2 - 1; i >= 0; i--)
				heapify(arr, indices, n, i);

			// One by one extract an element
			// from heap
			for (int i = n - 1; i > 0; i--) {
				// Move current root to end
				T temp = indices->get(0);
				indices->set(0, indices->get(i));
				indices->set(i, temp);

				// call max heapify on the reduced heap
				heapify(arr, indices, i, 0);
			}
		}
#pragma endregion
	}

	template <typename T>
	NDArray<T>* quicksort(NDArray<T>* a) {
		NDArray<T>* arr = a->copy();
		quicksort(arr, 0, a->getSize() - 1);
		return arr;
	}

	template <typename T>
	NDArray<T>* mergesort(NDArray<T>* a) {
		NDArray<T>* arr = a->copy();
		mergesort(arr, 0, a->getSize() - 1);
		return arr;
	}

	template <typename T>
	NDArray<T>* heapsort(NDArray<T>* a) {
		NDArray<T>* arr = a->copy();
		heapsort(arr, a->getSize());
		return arr;
	}

	template <typename T>
	void quicksort(NDArray<T>* a, NDArray<T>* indices) {
		quicksort(a, 0, a->getSize() - 1);
	}

	template <typename T>
	void mergesort(NDArray<T>* a, NDArray<T>* indices) {
		mergesort(a, indices, 0, a->getSize() - 1);
	}

	template <typename T>
	void heapsort(NDArray<T>* a, NDArray<T>* indices) {
		heapsort(a, indices, a->getSize());
	}

	template <typename T>
	NDArray<T>* sort(NDArray<T>* a, Kind kind = Quicksort) {
		switch (kind) {
		case Quicksort:
			return quicksort(a);
		case Mergesort:
			return mergesort(a);
		case Heapsort:
			return heapsort(a);
		default:
			return quicksort(a);
		}

		return nullptr;
	}

	template <typename T>
	Vector<T>* argsort(Vector<T>* arr, Kind kind = Quicksort) {
		Vector<T>* indices = Cudheart::VectorOps::arange(arr->getSize());

		switch (kind) {
		case Quicksort:
			quicksort(arr, indices);
		case Heapsort:
			heapsort(arr, indices);
		default:
			quicksort(arr, indices);
		}

		return indices;
	}

	template <typename T>
	Matrix<T>* argsort(Matrix<T>* arr, Kind kind = Quicksort) {
		Matrix<T>* indices = Cudheart::MatrixOps::arange(arr->getHeight(), arr->getWidth());

		switch (kind) {
		case Quicksort:
			quicksort(arr, indices);
		case Heapsort:
			heapsort(arr, indices);
		default:
			quicksort(arr, indices);
		}

		return indices;
	}

	template <typename T>
	NDArray<T>* partition(NDArray<T>* a, int kth) {
		NDArray<T>* out = a->emptyLike();

		T element = a->get(kth);
		int start = 0;
		int end = a->getSize() - 1;
		for (int i = 0; i < a->getSize(); i++) {
			if (i == kth) {
				continue;
			}
			T v = a->get(i);

			if (v < element) {
				out->set(start, v);
				start++;
			}
			else {
				out->set(end, v);
				end--;
			}
		}

		// start + 1 == end
		out->set(start + 1, element);

		return out;
	}

	template <typename T>
	Vector<T>* argpartition(Vector<T>* a, int kth) {
		Vector<T>* indices = Cudheart::VectorOps::arange(a->getSize());

		T element = a->get(kth);
		int start = 0;
		int end = a->getSize() - 1;
		for (int i = 0; i < a->getSize(); i++) {
			if (i == kth) {
				continue;
			}
			T v = a->get(i);

			if (v < element) {
				indices->set(start, i);
				start++;
			}
			else {
				indices->set(end, i);
				end--;
			}
		}

		// start + 1 == end
		indices->set(start + 1, kth);

		return indices;
	}

	template <typename T>
	Matrix<T>* argpartition(Matrix<T>* a, int kth) {
		Matrix<T>* indices = Cudheart::VectorOps::arange(a->getSize());

		T element = a->get(kth);
		int start = 0;
		int end = a->getSize() - 1;
		for (int i = 0; i < a->getSize(); i++) {
			if (i == kth) {
				continue;
			}
			T v = a->get(i);

			if (v < element) {
				indices->set(start, i);
				start++;
			}
			else {
				indices->set(end, i);
				end--;
			}
		}

		// start + 1 == end
		indices->set(start + 1, kth);

		return indices;
	}
}