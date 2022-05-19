#pragma once

#include "Cudheart/Arrays/Arrays.cuh"

#include <iostream>
#include <sstream>
#include <string>
#include <typeinfo>
#include <exception>
#include <climits>
#include <limits>
#include <stdio.h>
#include <stdarg.h>
#include <math.h>
#include <initializer_list>
#include <cstdarg>
#include <algorithm>
#include <array>
#include <iterator>
#include <string>

#pragma region
using std::exception;
using std::cout;
using std::cin;
using std::endl;
using std::string;
using std::ostringstream;
using std::ostream;
using std::initializer_list;
using std::to_string;
using std::array;
using std::copy;
using std::begin;
using std::end;
#pragma endregion

using namespace Cudheart::NDArrays;
using Cudheart::VectorOps::arange;
using namespace Cudheart::MatrixOps;

template <typename T>
NDArray<T>* add(NDArray<T>* a, NDArray<T>* b, int axis) {
	NDArray<T>* res;
	if (a->getDims() != b->getDims()) {
		if (a->getDims() == 1) {
			res = (NDArray<T>*)b->emptyLike();
			Vector<T>* v = (Vector<T>*)a;
			Matrix<T>* m = (Matrix<T>*)b;
			if (axis == 0) {
				for (int i = 0; i < m->getHeight(); i++) {
					for (int j = 0; j < m->getWidth(); j++) {
						res->set(i * m->getWidth() + j, v->getAbs(j) + m->get(i * m->getWidth() + j));
					}
				}
			}
			else if (axis == 1) {
				for (int i = 0; i < m->getHeight(); i++) {
					for (int j = 0; j < m->getWidth(); j++) {
						res->set(i * m->getWidth() + j, v->getAbs(i) + m->get(i * m->getWidth() + j));
					}
				}
			}
		}
		else {
			res = (NDArray<T>*)a->emptyLike();
			Vector<T>* v = (Vector<T>*)b;
			Matrix<T>* m = (Matrix<T>*)a;
			if (axis == 0) {
				for (int i = 0; i < m->getHeight(); i++) {
					for (int j = 0; j < m->getWidth(); j++) {
						res->set(i * m->getWidth() + j, v->getAbs(j) + m->get(i * m->getWidth() + j));
					}
				}
			}
			else if (axis == 1) {
				for (int i = 0; i < m->getHeight(); i++) {
					for (int j = 0; j < m->getWidth(); j++) {
						res->set(i * m->getWidth() + j, v->getAbs(i) + m->get(i * m->getWidth() + j));
					}
				}
			}
		}
	}
	else {
		res = a->emptyLike();

		for (int i = 0; i < a->getSize(); i++) {
			res->set(i, a->get(i) + b->get(i));
		}
	}

	return res;
}

template <typename T>
NDArray<T>* add(NDArray<T>* a, NDArray<T>* b) {
	return add<T>(a, b, 0);
}

void testThing() {
	Vector<int>* a = Cudheart::VectorOps::arange<int>(1, 12, 2);
	//print(a);
	Matrix<int>* b = Cudheart::MatrixOps::full(6, 6, 5);
	//print(b);
	//print((Vector<int>*)add(a, a));
	//print((Matrix<int>*)add(b, b));
	((Matrix<int>*)(add(a, b, 0)))->print();
	((Matrix<int>*)(add(a, b, 1)))->print();
	//print((Matrix<int>*)add(b, a, 0));
	//print((Matrix<int>*)add(b, a, 1));
}