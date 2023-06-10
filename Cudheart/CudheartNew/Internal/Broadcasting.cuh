#pragma once

#include "../Arrays/NDArray.cuh"

#include <algorithm>
#include <tuple>

namespace CudheartNew {
	std::vector<NDArrayBase*> broadcast(std::vector<NDArrayBase*> const& arrays) {
		std::vector<long> finalShape;

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
					fmt::println("cannot broadcast these arrays together");
					exit(-1);
				}
			}
		}

		std::vector<NDArrayBase*> out(arrays.size());
		std::transform(arrays.begin(), arrays.end(), out.begin(), [&finalShape](NDArrayBase* arr) {
			return arr->broadcastTo(finalShape);
			});

		return out;
	}
}