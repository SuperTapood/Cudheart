#pragma once


namespace Cudheart::Arrays {
	template <typename T>
	class ArrayOps {
	public:
		// do not fucking use this with T = char, string or custom class
		static Array<T>* arange(T start, T end, T jump) {
			int len = (int)((end - start) / jump);
			Array<T>* out = empty(new Shape(new int[]{len}, 1));
			
			for (int i = 0; start < end; start += jump) {
				out->data[i++] = start;
			}

			return out;
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

		static Array<T>* eye(int rows, int cols, int k)
		{
			Array<T>* out = zeros(new Shape(new int[] {rows, cols}, 2));

			for (int pos = 0; k < cols; k++) {
				for (int i = 0; i < rows; i++, pos++) {
					k = k % cols;
					out->data[pos] = 1;
				}
			}

			return out;
		}


		static Array<T>* eye(int rows) {
			return eye(rows, rows, 0);
		}


		static Array<T>* eye(int rows, int k) {
			return eye(rows, rows, k);
		}

		static Array<T>* full(Shape* shape, T value) {
			Array<T>* out = empty(new Shape(new int[] {shape->size}, 1));

			for (int i = 0; i < shape->size; i++) {
				out->data[i] = value;
			}

			return (*out).reshape(shape);
		}

		static Array<T>* ones(Shape* shape) {
			return full(shape, 1);
		}

		static Array<T>* zeros(Shape* shape) {
			return full(shape, 0);
		}
	};
}
