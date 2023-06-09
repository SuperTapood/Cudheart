#pragma once

#include "../Arrays/NDArray.cuh"

#include <algorithm>
#include <tuple>

namespace CudheartNew {
	std::vector<NDArrayBase*> broadcast(std::vector<NDArrayBase*> const& arrays) {
		std::vector<long> finalShape = arrays.at(0)->shape();

		for (auto arr : arrays) {
			long diff = (long)arr->ndims() - (long)finalShape.size();

			for (int i = 0; i < diff; i++) {
				finalShape.insert(finalShape.begin(), 1);
			}

			for (int i = 0; i < arr->ndims(); i++) {
				if (arr->shape().at(arr->ndims() - 1 - i) == 1) {
					continue;
				}
				else if (finalShape.at(finalShape.size() - 1 - i) == 1) {
					finalShape.at(finalShape.size() - 1 - i) = arr->shape().at(arr->ndims() - 1 - i);
				}
				else if (finalShape.at(finalShape.size() - 1 - i) != arr->shape().at(arr->ndims() - 1 - i)) {
					fmt::println("cannot cast these arrays together");
					exit(-1);
				}
			}
		}

		/*auto minIndex = std::distance(arrays.begin(), std::min_element(arrays.begin(), arrays.end(), [](NDArrayBase const* a, NDArrayBase const* b)
			{
				return a->ndims() < b->ndims();
			}));

		auto min = arrays.at(minIndex);

		for (int i = 0; i < arrays.size(); i++) {
			if (i == minIndex) {
				continue;
			}

			min = min->broadcastTo(arrays.at(i)->shape());
		}*/

		std::vector<NDArrayBase*> out(arrays.size());
		std::transform(arrays.begin(), arrays.end(), out.begin(), [&finalShape](NDArrayBase* arr) {
			return arr->broadcastTo(finalShape);
			});

		return out;
	}
}