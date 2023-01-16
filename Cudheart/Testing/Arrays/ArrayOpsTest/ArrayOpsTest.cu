#include "ArrayOpsTest.cuh"

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
		check("ArrayOps::append(Vector<int>*, int)", append(vec, 4)->toString());
		check("ArrayOps::append(Matrix<int>*, Vector<int>*)", append(mat, vec)->toString());
	}

	void testConcatenate() {
		using namespace Cudheart::ArrayOps;
		using namespace Cudheart::MatrixOps;
		using namespace Cudheart::IO;
		string str = "1 2 3";
		string res = "[1, 2, 3, 1, 2, 3]";
		auto a = fromString<float>(str);
		auto b = fromString<float>(str);

		check("ArrayOps::concatenate(Vector<float>, Vector<float>)", concatenate(a, b)->toString());

		string matres = "[\n [0, 1],\n [2, 3],\n [0, 1],\n [2, 3]\n]";

		auto c = arange(4, 2, 2);
		auto d = arange(4, 2, 2);

		check("ArrayOps::concatenate(Matrix<int>, Matrix<int>)", concatenate(c, d)->toString());
	}

	void testSplit() {
		using namespace Cudheart::VectorOps;
		using namespace Cudheart::ArrayOps;

		int size = 15;
		int vecs = 5;

		auto a = arange(size);
		auto arr = split(a, vecs);

		check("ArrayOps::split(Vector<int>, int)", arr->toString());

		auto b = (Matrix<int>*)a->reshape<int>(new Shape((int)(size / vecs), vecs));
		auto brr = split(b, size / vecs);

		check("ArrayOps::split(Matrix<int>, int)", brr->toString());
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

		check("ArrayOps::tile(Vector<int>, int)", arr->toString());

		auto brr = tile(a, wReps, hReps);

		check("ArrayOps::tile(Vector<int>, int, int)", brr->toString());

		auto c = (Matrix<int>*)(a->reshape<int>(new Shape(h, w)));
		auto crr = tile(c, reps);

		check("ArrayOps::tile(Matrix<int>, int)", crr->toString());

		auto drr = tile(c, wReps, hReps);

		check("ArrayOps::tile(Matrix<int>, int, int)", drr->toString());
	}

	void testRemove() {
		using namespace Cudheart::VectorOps;
		using namespace Cudheart::ArrayOps;

		int size = 20;
		int rem = 3;

		auto a = arange(size);
		auto arr = remove(a, rem);
		int idx = 0;

		check("ArrayOps::remove(Vector<int>, int)", arr->toString());

		auto b = (Matrix<int>*)a->reshape<int>(new Shape(4, 5));
		auto brr = remove(b, rem);

		check("ArrayOps::remove(Matrix<int>, int, axis=-1)", brr->toString());

		auto crr = (Matrix<int>*)(remove<int>(b, rem, 0));

		check("ArrayOps::remove(Matrix<int>, int, axis=0)", crr->toString());

		auto drr = (Matrix<int>*)(remove<int>(b, rem, 1));

		check("ArrayOps::remove(Matrix<int>, int, axis=1)", drr->toString());
	}

	void testTrimZeros() {
		using namespace Cudheart::VectorOps;
		using namespace Cudheart::ArrayOps;
		using namespace Cudheart::IO;

		auto a = fromString<int>("0, 0, 0, 1, 2, 3, 0, 2, 1, 0");

		check("ArrayOps::trimZeros(Vector<int>, 'fb')", trimZeros(a)->toString());
		check("ArrayOps::trimZeros(Vector<int>, 'f')", trimZeros(a, "f")->toString());
		check("ArrayOps::trimZeros(Vector<int>, 'b')", trimZeros(a, "b")->toString());
	}

	void testUnique() {
		using namespace Cudheart::VectorOps;
		using namespace Cudheart::ArrayOps;
		using namespace Cudheart::IO;

		auto a = fromString<int>("0, 0, 0, 1, 3, 2, 0, 2, 1, 0");

		Vector<int>* res;

		res = unique(a, false, false, false)->get(0);
		check("ArrayOps::unique(a, false, false, false)", res->toString());

		res = unique(a, true, false, false)->get(1);
		check("ArrayOps::unique(a, true, false, false)", res->toString());

		res = unique(a, false, true, false)->get(2);
		check("ArrayOps::unique(a, false, true, false)", res->toString());

		res = unique(a, false, false, true)->get(3);
		check("ArrayOps::unique(a, false, false, true)", res->toString());
	}
}