#pragma once


namespace Cudheart::Arrays {
	template <typename T>
	class ArrayOps {
	public:
		// do not fucking use this with T = char, string or custom class
		static Array<T>* arange(T start, T end, T jump) {
			int len = (int)((end - start) / jump);
			Array<T>* out = empty(new Shape({len}, 1);
			
			for (int i = 0; start < end; start += jump) {
				out->data[i++] = start;
			}
		}

		static Array<T>* arange(T end, T jump) {
			return arange(0, end, jump);
		}

		static Array<T>* arange(T end) {
			return arange(0, end, 1);
		}

		static Array<T>* empty(Shape* shape) {
			return new Array<T>(shape);
		}

		static Array<T>* emptyLike(Array<T>* arr) {
			return empty(arr->shape);
		}

		static Array<T>* full(Shape* shape, T value) {
			Array<T>* out = empty(shape->size);
			
			for (int i = 0; i < shape->size; i++) {
				out->data[i] = value;
			}

			return out.reshape(shape);
		}
	};
}
