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
}