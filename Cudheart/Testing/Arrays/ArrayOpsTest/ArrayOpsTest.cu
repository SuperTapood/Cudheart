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
		string matres = "[\n [0, 1, 2],\n [3, 4, 5],\n [6, 7, 8],\n [1, 2, 3]\n]";
		string cmd;

		auto vec = fromString<int>(str);
		
		cmd =  Numpy::arange("1", "4", "1", "int", "vec");
		cmd += Numpy::append("vec", "4", "None", "res");

		auto a = append(vec, 4);

		Testing::check("ArrayOps::append(Vector<int>*, int)", cmd, a->toString());

		auto mat = arange(9, 3, 3);

		auto b = append(mat, vec);

		cmd = Numpy::arange("1", "4", "1", "int", "vec");
		cmd += Numpy::arange("0", "9", "1", "int", "mat");
		cmd += Numpy::append("mat", "vec", "None", "res");
		cmd += Numpy::reshape("res", "(4, 3)", "res");

		Testing::check("ArrayOps::append(Matrix<int>*, Vector<int>*)", cmd, b->toString());
	}

	void testConcatenate() {
		using namespace Cudheart::ArrayOps;
		using namespace Cudheart::MatrixOps;
		using namespace Cudheart::IO;
		string cmd;

		string str = "1 2 3";
		string res = "[1, 2, 3, 1, 2, 3]";
		auto a = fromString<float>(str);
		auto b = fromString<float>(str);

		cmd =  Numpy::arange("1", "4", "1", "int", "a");
		cmd += Numpy::arange("1", "4", "1", "int", "b");
		cmd += Numpy::concatenate("a", "b", "res");

		Testing::check("ArrayOps::concatenate(Vector<float>, Vector<float>)", cmd, concatenate(a, b)->toString());

		string matres = "[\n [0, 1],\n [2, 3],\n [0, 1],\n [2, 3]\n]";

		auto c = arange(4, 2, 2);
		auto d = arange(4, 2, 2);

		cmd =  Numpy::arange("0", "4", "1", "int", "a");
		cmd += Numpy::reshape("a", "(2, 2)", "a");
		cmd += Numpy::arange("0", "4", "1", "int", "b");
		cmd += Numpy::reshape("b", "(2, 2)", "b");
		cmd += Numpy::concatenate("a", "b", "res");

		Testing::check("ArrayOps::concatenate(Matrix<int>, Matrix<int>)", cmd, concatenate(c, d)->toString());
	}

	void testSplit() {
		using namespace Cudheart::VectorOps;
		using namespace Cudheart::ArrayOps;
		string cmd;

		int size = 15;
		int vecs = 5;

		auto a = arange(size);
		auto arr = split(a, vecs);

		cmd =  Numpy::arange("0", to_string(size), "1", "int", "a");
		cmd += Numpy::split("a", to_string(vecs), "res");
		cmd += "comp = np.array([";
		for (int i = 0; i < vecs - 1; i++) {
			cmd += arr[i]->toString() + ", ";
		}
		cmd += arr[vecs - 1]->toString() + "])";

		Testing::check("ArrayOps::split(Vector<int>, int)", cmd, "comp");

		auto b = (Matrix<int>*)a->reshape<int>(new Shape((int)(size / vecs), vecs));
		auto brr = split(b, vecs);

		Testing::check("ArrayOps::split(Matrix<int>, int)", cmd, "comp");
	}

	void testTile() {
		using namespace Cudheart::VectorOps;
		using namespace Cudheart::ArrayOps;
		string cmd;

		int w = 5;
		int h = 3;
		int size = w * h;
		int reps = 5;
		int hReps = 2;
		int wReps = 2;

		auto a = arange(size);
		auto arr = tile(a, reps);

		cmd =  Numpy::arange("0", to_string(size), "1", "int", "a");
		cmd += Numpy::tile("a", to_string(reps), "res");

		Testing::check("ArrayOps::tile(Vector<int>, int)", cmd, arr->toString());

		auto brr = tile(a, hReps, wReps);

		cmd = Numpy::arange("0", to_string(size), "1", "int", "a");
		cmd += Numpy::tile("a", "(" + to_string(hReps) + ", " + to_string(wReps) + ")", "res");

		Testing::check("ArrayOps::tile(Vector<int>, int, int)", cmd, brr->toString());

		auto c = (Matrix<int>*)(a->reshape<int>(new Shape(h, w)));
		auto crr = tile(c, reps);

		cmd =  Numpy::arange("0", to_string(size), "1", "int", "a");
		cmd += Numpy::reshape("a", "(" + to_string(h) + ", " + to_string(w) + ")", "a");
		cmd += Numpy::tile("a", to_string(reps), "res");

		Testing::check("ArrayOps::tile(Matrix<int>, int)", cmd, crr->toString());

		auto drr = tile(c, wReps, hReps);

		cmd = Numpy::arange("0", to_string(size), "1", "int", "a");
		cmd += Numpy::reshape("a", "(" + to_string(h) + ", " + to_string(w) + ")", "a");
		cmd += Numpy::tile("a", "(" + to_string(hReps) + ", " + to_string(wReps) + ")", "res");

		Testing::check("ArrayOps::tile(Matrix<int>, int, int)", cmd, drr->toString());
	}

	void testRemove() {
		using namespace Cudheart::VectorOps;
		using namespace Cudheart::ArrayOps;
		string cmd;

		int size = 20;
		int rem = 3;

		auto a = arange(size);
		auto arr = remove(a, rem);
		int idx = 0;

		cmd = Numpy::arange("0", to_string(size), "1", "int", "a");
		cmd += Numpy::remove("a", to_string(rem), "res");

		Testing::check("ArrayOps::remove(Vector<int>, int)", cmd, arr->toString());

		auto b = (Matrix<int>*)a->reshape<int>(new Shape(4, 5));
		auto brr = remove(b, rem);

		cmd = Numpy::arange("0", to_string(size), "1", "int", "a");
		cmd += Numpy::remove("a", to_string(rem), "res");

		Testing::check("ArrayOps::remove(Matrix<int>, int, axis=-1)", cmd, brr->toString());

		auto crr = (Matrix<int>*)(remove<int>(b, rem, 0));

		cmd = Numpy::arange("0", to_string(size), "1", "int", "a");
		cmd += Numpy::reshape("a", "(4, 5)", "a");
		cmd += Numpy::remove("a", to_string(rem), "0", "res");

		Testing::check("ArrayOps::remove(Matrix<int>, int, axis=0)", cmd, crr->toString());

		auto drr = (Matrix<int>*)(remove<int>(b, rem, 1));

		cmd = Numpy::arange("0", to_string(size), "1", "int", "a");
		cmd += Numpy::reshape("a", "(4, 5)", "a");
		cmd += Numpy::remove("a", to_string(rem), "1", "res");

		Testing::check("ArrayOps::remove(Matrix<int>, int, axis=1)", cmd, drr->toString());
	}

	void testTrimZeros() {
		using namespace Cudheart::VectorOps;
		using namespace Cudheart::ArrayOps;
		using namespace Cudheart::IO;

		auto a = fromString<int>("0, 0, 0, 1, 2, 3, 0, 2, 1, 0");

		string cmd = Numpy::createArray("[0, 0, 0, 1, 2, 3, 0, 2, 1, 0]", "a");
		cmd += Numpy::trim_zeros("a", "'fb'", "res");

		Testing::check("ArrayOps::trimZeros(Vector<int>, 'fb')", cmd, trimZeros(a)->toString());

		cmd = Numpy::createArray("[0, 0, 0, 1, 2, 3, 0, 2, 1, 0]", "a");
		cmd += Numpy::trim_zeros("a", "'f'", "res");

		Testing::check("ArrayOps::trimZeros(Vector<int>, 'f')", cmd, trimZeros(a, "f")->toString());

		cmd = Numpy::createArray("[0, 0, 0, 1, 2, 3, 0, 2, 1, 0]", "a");
		cmd += Numpy::trim_zeros("a", "'b'", "res");

		Testing::check("ArrayOps::trimZeros(Vector<int>, 'b')", cmd, trimZeros(a, "b")->toString());
	}

	void testUnique() {
		using namespace Cudheart::VectorOps;
		using namespace Cudheart::ArrayOps;
		using namespace Cudheart::IO;
		string cmd, add;

		auto a = fromString<int>("0, 0, 0, 1, 3, 2, 0, 2, 1, 0");

		auto res = unique(a, true, true, true);

		cmd = Numpy::createArray("[0, 0, 0, 1, 3, 2, 0, 2, 1, 0]", "a");
		cmd += Numpy::unique("a", "res");
		cmd += "comp = [";
		for (int i = 0; i < 3; i++) {
			cmd += res[i]->toString() + ", ";
		}
		cmd += res[3]->toString() + "]\n";

		/*res[0]->print();
		res[1]->print();
		res[2]->print();
		res[3]->print();*/

		add = "res = res[0]";

		Testing::check("ArrayOps::unique(a, false, false, false)", cmd + add, "comp[0]");

		add = "res = res[1]";

		Testing::check("ArrayOps::unique(a, true, false, false)", cmd + add, "comp[1]");

		add = "res = res[2]";

		Testing::check("ArrayOps::unique(a, false, true, false)", cmd + add, "comp[2]");

		add = "res = res[3]";

		Testing::check("ArrayOps::unique(a, false, false, true)", cmd + add, "comp[3]");
	}
}