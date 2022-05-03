#pragma once

#include "../Arrays/Arrays.cuh"

namespace Cudheart::CPP::Logic {
	template <typename T>
	bool all(Vector<T>* vec) {
		for (int i = 0; i < vec->getSize(); i++) {
			if (!(bool)vec->get(i)) {
				return false;
			}
		}
		
		return true;
	}

	template <typename T>
	bool all(Matrix<T>* mat) {
		for (int i = 0; i < mat->getSize(); i++) {
			if (!(bool)mat->get(i)) {
				return false;
			}
		}

		return true;
	}

	template <typename T>
	bool any(Vector<T>* vec) {
		for (int i = 0; i < vec->getSize(); i++) {
			if ((bool)vec->get(i)) {
				return true;
			}
		}

		return false;
	}

	template <typename T>
	bool any(Matrix<T>* mat) {
		for (int i = 0; i < mat->getSize(); i++) {
			if ((bool)mat->get(i)) {
				return true;
			}
		}

		return false;
	}
}