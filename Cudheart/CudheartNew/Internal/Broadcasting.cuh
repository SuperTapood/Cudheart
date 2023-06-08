#pragma once

#include "../Arrays/NDArray.cuh"

#include <algorithm>
#include <tuple>

namespace CudheartNew {
	template <typename ... Ts>
	std::vector<NDArrayBase*> broadcast(std::vector<NDArrayBase*> const& arrays) {
		auto minIndex = std::distance(arrays.begin(), std::min_element(arrays.begin(), arrays.end(), [](NDArrayBase* a, NDArrayBase* b)
			{
			return a->ndims() < b->ndims();
			}));

		auto min = arrays.at(minIndex);

		for (int i = 0; i < arrays.size(); i++) {
			if (i == minIndex) {
				continue;
			}

			min = min->broadcastTo(arrays.at(i)->shape());
		}

		std::vector<NDArrayBase*> out(arrays.size());
		std::transform(arrays.begin(), arrays.end(), out.begin(), [min](NDArrayBase* arr) {
			return arr->broadcastTo(min->shape());
			});

		return out;
	}
}