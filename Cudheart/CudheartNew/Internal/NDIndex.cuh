#pragma once

#include <vector>


namespace CudheartNew {
	std::vector<std::vector<long>> ndindex(std::vector<long> shape) {
		std::vector<std::vector<long>> out;

		std::vector<long> s(shape.size(), 0);

		long size = 1;

		for (auto val : shape) {
			size *= val;
		}

		out.reserve(size);

		for (int i = 0; i < size; i++) {
			out.push_back(s);

			for (int j = s.size() - 1; j >= 0; j--) {
				s.at(j)++;

				if (s.at(j) == shape.at(j)) {
					s.at(j) = 0;
				}
				else {
					break;
				}
			}
		}

		return out;
	}
}