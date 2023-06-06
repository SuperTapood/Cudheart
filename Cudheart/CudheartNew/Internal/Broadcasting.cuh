#pragma once

#include "../Arrays/NDArray.cuh"

#include <algorithm>
#include <tuple>

namespace CudheartNew {
	template <typename A, typename B>
	std::pair<NDArray<A>*, NDArray<B>*> broadcast(NDArray<A>* a, NDArray<B>* b) {
		auto res = broadcast(std::make_tuple(a, b));

		return { std::get<0>(res), std::get<1>(res) };
	}

	template <typename ... Ts>
	std::tuple<NDArray<Ts>*...> broadcast(std::tuple<NDArray<Ts>*...> arrays) {
		// todo: broadcast the first of the arrays with the rest of the arrays and then brodcast everyone to its shape since it now has the "final shape" they all should have
		auto shape = maxShape(arrays);

		return std::apply([&shape](auto&... args) {
			return std::make_tuple(args->broadcastTo(shape)...);
			}, arrays);
	}

	template <typename ... Ts>
	auto maxShape(std::tuple<NDArray<Ts>*...> arrays) {
		return std::apply([](auto&&... args) {
			return std::max({ args... }, [](const auto& a, const auto& b) {
				return a->ndims() < b->ndims();
				});
			}, arrays);
	}
}