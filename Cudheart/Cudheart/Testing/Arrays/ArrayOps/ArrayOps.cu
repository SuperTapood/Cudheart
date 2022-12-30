#include "ArrayOps.cuh"

namespace Cudheart::Testing::Arrays::ArrayOps {
	void test() {
		testAppend();
		testConcatenate();
		testSplit();
		testTile();
		testRemove();
		testTrimZeros();
		testUnique();
	}

	void testAppend() {
		using namespace Cudheart::ArrayOps;
		using namespace Cudheart::MatrixOps;
		using namespace Cudheart::IO;
		string str = "1 2 3";
		string res = "[1, 2, 3, 4]";
		string matres = "[\n [0, 1, 2],\n [3, 4, 5],\n [6, 7, 8],\n [1, 2, 3]\n]";
		auto vec = fromString<int>(str);
		auto mat = arange(9, 3, 3);
		check("append(Vector<int>*, int)", append(vec, 4)->toString() == res);
		check("append(Matrix<int>*, Vector<int>*)", append(mat, vec)->toString() == matres);
	}

	void testConcatenate() {
		using namespace Cudheart::ArrayOps;
		using namespace Cudheart::MatrixOps;
		using namespace Cudheart::IO;
		string str = "1 2 3";
		string res = "[1, 2, 3, 1, 2, 3]";
		auto a = fromString<float>(str);
		auto b = fromString<float>(str);

		check("concatenate(Vector<float>, Vector<float>)", concatenate(a, b)->toString() == res);

		string matres = "[\n [0, 1],\n [2, 3],\n [0, 1],\n [2, 3]\n]";

		auto c = arange(4, 2, 2);
		auto d = arange(4, 2, 2);

		check("concatenate(Matrix<int>, Matrix<int>)", concatenate(c, d)->toString() == matres);
	}

	void testSplit() {
		using namespace Cudheart::VectorOps;
		using namespace Cudheart::ArrayOps;

		int size = 15;
		int vecs = 5;

		auto a = arange(size);
		auto arr = split(a, vecs);

		for (int i = 0; i < vecs; i++) {
			for (int j = 0; j < size / vecs; j++) {
				check("split(Vector<int>, int)", arr[i]->get(j) == a->get(i * (size / vecs) + j));
			}
		}

		auto b = (Matrix<int>*)a->reshape<int>(new Shape((int)(size / vecs), vecs));
		auto brr = split(b, size / vecs);

		for (int i = 0; i < size / vecs; i++) {
			for (int j = 0; j < vecs; j++) {
				check("split(Matrix<int>, int)", brr[i]->get(j) == b->get(i * vecs + j));
			}
		}
	}

	void testTile() {
		using namespace Cudheart::VectorOps;
		using namespace Cudheart::ArrayOps;

		int w = 5;
		int h = 3;
		int size = w * h;
		int reps = 5;
		int hReps = 2;
		int wReps = 2;

		auto a = arange(size);
		auto arr = tile(a, reps);

		for (int i = 0; i < reps; i++) {
			for (int j = 0; j < size; j++) {
				check("tile(Vector<int>, int)", arr->get(i * size + j) == a->get(j));
			}
		}

		auto brr = tile(a, wReps, hReps);

		for (int i = 0; i < hReps; i++) {
			for (int j = 0; j < wReps; j++) {
				for (int k = 0; k < a->getSize(); k++) {
					check("tile(Vector<int>, int, int)", brr->get(i, k + a->getSize() * j) == a->get(k));
				}
			}
		}

		auto c = (Matrix<int>*)(a->reshape<int>(new Shape(h, w)));
		auto crr = tile(c, reps);

		for (int i = 0; i < h; i++) {
			for (int j = 0; j < reps; j++) {
				for (int k = 0; k < w; k++) {
					check("tile(Matrix<int>, int)", crr->get(i, k + w * j) == c->get(i, k));
				}
			}
		}

		auto drr = tile(c, wReps, hReps);

		for (int i = 0; i < hReps; i++) {
			for (int j = 0; j < h; j++) {
				for (int k = 0; k < wReps; k++) {
					for (int m = 0; m < w; m++) {
						check("tile(Matrix<int>, int, int)", drr->get(j + i * h, m + k * w) == c->get(j, m));
					}
				}
			}
		}
	}

	void testRemove() {
		using namespace Cudheart::VectorOps;
		using namespace Cudheart::ArrayOps;

		int size = 20;
		int rem = 3;

		auto a = arange(size);
		auto arr = remove(a, rem);
		int idx = 0;

		for (int i = 0; i < size; i++) {
			if (i != rem) {
				check("remove(Vector<int>, int)", i == arr->get(idx));
				idx++;
			}

		}

		auto b = (Matrix<int>*)a->reshape<int>(new Shape(4, 5));
		auto brr = remove(b, rem);

		for (int i = 0; i < size - 1; i++) {
			check("remove(Matrix<int>, int, axis=-1)", brr->get(i) == arr->get(i));
		}

		auto crr = (Matrix<int>*)(remove<int>(b, rem, 0));

		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 5; j++) {
				check("remove(Matrix<int>, int, axis=0)", crr->get(i, j) == b->get(i, j));
			}
		}


		auto drr = (Matrix<int>*)(remove<int>(b, rem, 1));
	}

	void testTrimZeros() {
		using namespace Cudheart::VectorOps;
		using namespace Cudheart::ArrayOps;
		using namespace Cudheart::IO;

		auto a = fromString<int>("0, 0, 0, 1, 2, 3, 0, 2, 1, 0");

		check("trimZeros(Vector<int>, 'fb')", trimZeros(a)->toString() == "[1, 2, 3, 0, 2, 1]");
		check("trimZeros(Vector<int>, 'f')", trimZeros(a, "f")->toString() == "[1, 2, 3, 0, 2, 1, 0]");
		check("trimZeros(Vector<int>, 'b')", trimZeros(a, "b")->toString() == "[0, 0, 0, 1, 2, 3, 0, 2, 1]");
	}

	void testUnique() {
		using namespace Cudheart::VectorOps;
		using namespace Cudheart::ArrayOps;
		using namespace Cudheart::IO;

		auto a = fromString<int>("0, 0, 0, 1, 3, 2, 0, 2, 1, 0");

		Vector<int>* res;


		res = unique(a, false, false, false)[0];
		check("unique(a, false, false, false)", res->toString() == "[0, 1, 3, 2]");

		res = unique(a, true, false, false)[1];
		check("unique(a, true, false, false)", res->toString() == "[0, 3, 4, 5]");

		res = unique(a, false, true, false)[2];
		check("unique(a, false, true, false)", res->toString() == "[0, 0, 0, 1, 2, 3, 0, 3, 1, 0]");

		res = unique(a, false, false, true)[3];
		check("unique(a, false, false, true)", res->toString() == "[5, 2, 1, 2]");
	}
}