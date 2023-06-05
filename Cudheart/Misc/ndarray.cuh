#pragma once
#include <stdio.h>
#include <initializer_list>
#include <iostream>
#include <vector>
#include <type_traits>
#define FMT_HEADER_ONLY
#include <fmt/ranges.h>
#include <memory>

#include <new>

#include "../CudheartNew/Arrays/NDArray.cuh"
#include "../CudheartNew/Arrays/ArrayOps.cuh"

namespace CudheartNew {
	template <typename A, typename B>
	std::pair<NDArray<A>*, NDArray<B>*> broadcast(NDArray<A>* a, NDArray<B>* b) {
		a = a->copy();
		b = b->copy();

		int diff = a->ndims() - b->ndims();
		std::vector<int> newShape;

		for (int i = 0; i < std::abs(diff); i++) {
			newShape.push_back(1);
		}

		if (a->ndims() > b->ndims()) {
			for (int i = 0; i < b->ndims(); i++) {
				newShape.push_back(b->shape().at(i));
			}
			b->reshape(newShape, true);
		}
		else if (b->ndims() > a->ndims()) {
			for (int i = 0; i < a->ndims(); i++) {
				newShape.push_back(a->shape().at(i));
			}
			a->reshape(newShape, true);
		}

		newShape.clear();

		for (int i = 0; i < a->ndims(); i++) {
			auto sa = a->shape().at(i);
			auto sb = b->shape().at(i);

			if (sa == sb) {
				newShape.push_back(sa);
				continue;
			}

			if (sa == 1) {
				newShape.push_back(sb);
				continue;
			}
			else if (sb == 1) {
				newShape.push_back(sa);
				continue;
			}

			fmt::println("shapes {} and {} cannot be broadcasted", a->shapeString(), b->shapeString());
			exit(-1);
		}

		//fmt::println("final shape: ({})", fmt::join(newShape, ","));

		a = a->stretch(newShape);
		b = b->stretch(newShape);

		return { a, b };
	}

	template <typename A, typename B, typename T = promote(A, B)>
	NDArray<T>* add(NDArray<A>* x, NDArray<B>* y) {
		auto [a, b] = broadcast(x, y);

		auto result = new NDArray<T>(a->shape());

#pragma omp parallel for
		for (int i = 0; i < result->size(); i++) {
			result->at(i) = (T)a->at(i) + (T)b->at(i);
		}

		return result;
	}

	template <typename A>
	A sum(NDArray<A>* x) {
		A total = 0;

		for (int i = 0; i < x->size(); i++) {
			total += x->at(i);
		}

		return total;
	}

	template <typename A>
	NDArray<A>* sum(NDArray<A>* x, int axis) {
		auto result = new NDArray<A>(x->m_shape->subshape(axis));

		int index = 0;

		for (int idx = 0; idx < x->m_shape->subsize(axis); idx++) {
			auto indices = x->getAxis(axis, idx);
			result->m_data[index++] = sum(x->subarray(indices));
		}

		return result;
	}


	void test() {
		using namespace CudheartNew::ArrayOps;
		auto a = new NDArray<int>({ 1, 5, 5 });
		auto b = new NDArray<int>({ 3, 5, 5 });

		for (int i = 0; i < a->size(); i++) {
			a->at(i) = i;
		}

		for (int i = 0; i < b->size(); i++) {
			b->at(i) = i;
		}

		/*b->println();

		add(a, b);*/

		// b->rot90(0, 1)->println();

		/*for (auto s : CudheartNew::ArrayOps::ndindex(a->shape())) {
			fmt::println("{}", fmt::join(s, ","));
		}*/

		std::vector<NDArray<int>*> vec = { a, b };

		auto out = CudheartNew::ArrayOps::unique(a);

		out.at(0)->println();

		//b->println();
		//fmt::println("{}", b->getAxis(0, 0));

		//a->stretch(b->m_shape)->println();

		/*auto [ra, rb] = broadcast(a, b);

		ra->println();
		rb->println();
		fmt::println("a - {}\nb - {}", a->m_shape->toString(), b->m_shape->toString());*/

		/*auto r = add(a, b);

		fmt::println("{}", sum(a));

		sum(b, 0)->println();*/

		/*delete r;
		delete a;
		delete b;
		delete ra;
		delete rb;*/
	}
}