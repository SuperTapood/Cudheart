#pragma once

#include "../Util.cuh"


namespace Cudheart::NDArrays {
	template <typename T>
	class NDArray {
	public:
		virtual ~NDArray() {}
		virtual T get(int index) = 0;
		virtual void set(int index, T value) = 0;
		virtual string toString() = 0;
		virtual void print() = 0;
	};
}