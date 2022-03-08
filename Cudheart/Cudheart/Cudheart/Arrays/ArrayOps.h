#pragma once


namespace Cudheart::Arrays {
	template <typename T>
	class ArrayOps {
	public:
		static Array<T>* asarray(T* arr, Shape* shape) {
			return new Array<T>(arr, shape->clone());
		}
		// do not fucking use this with T = char, string or custom class
		// if you do, prepare for trouble and make it double
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
			return empty(arr->shape->clone());
		}

		static Array<T>* eye(int rows, int cols, int k)
		{
			Array<T>* out = zeros(new Shape(new int[] {rows, cols}, 2));

			for (int i = 0; k < cols && i < rows; i++, k++) {
				out->data[i * rows + k] = 1;
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

		static Array<T>* fullLike(Array<T>* arr, T value) {
			return full(arr->shape->clone(), value);
		}

		static Array<T>* linspace(T start, T stop, T num, bool endpoint) {
			T jump = (stop - start) / num;
			if (endpoint) {
				jump = (stop - start) / (num - 1);
				stop += jump;
			}
			return arange(start, stop, jump);
		}

		static Array<T>* linspace(T start, T stop, T num) {
			return linspace(start, stop, num, true);
		}

		static Array<T>* linspace(T start, T stop) {
			return linspace(start, stop, 50, true);
		}

		static Array<T>* meshgrid(Array<T>* a, Array<T>* b)
		{
			if ((*a).shape->len != 1 || (*b).shape->len != 1) {
				return nullptr;
			}
			int alen = (*a).shape->dims[0];
			int blen = (*b).shape->dims[0];
			Array<T>* out = (Array<T>*)malloc(sizeof(Array<T>) * 2);
			out[0] = *zeros(new Shape(new int[] {blen, alen}, 2));
			out[1] = *zeros(new Shape(new int[] {alen, blen}, 2));

			print(out[0]);
			print(out[1]);

			for (int i = 0; i < blen; i++) {
				for (int j = 0; j < alen; j++) {
					Array<T> v = out[0];
					v.data[i * blen + j] = (*a).data[j];
				}
			}

			for (int i = 0; i < alen; i++) {
				for (int j = 0; j < blen; j++) {
					Array<T> v = out[1];
					v.data[j * alen + i] = (*b).data[j];
				}
			}

			return out;
		}

		static Array<T>* ones(Shape* shape) {
			return full(shape, 1);
		}

		static Array<T>* onesLike(Array<T>* arr) {
			return ones(arr->shape->clone());
		}

		static Array<T>* zeros(Shape* shape) {
			return full(shape, 0);
		}

		static Array<T>* zerosLike(Array<T>* arr) {
			return zeros(arr->shape->clone());
		}

		
	};
}
