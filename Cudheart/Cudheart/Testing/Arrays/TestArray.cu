#include "TestArray.cuh" 

namespace Cudheart::Testing {
	void testArrays() {
		auto start = std::chrono::system_clock::now();
		// tests are sorted in a hierarchical order
		// dependencies will be tested first, then the modules that depend on them
		// this is done to simplify debugging upon feature breaking
		Arrays::VecTest::test();
		Arrays::IO::test();
		Arrays::ArrayOps::test();
		auto end = std::chrono::system_clock::now();
		std::chrono::duration<double> elapsed = end - start;
		// cout << "array tests passed in " << elapsed.count() << "s\n";	
	}
}

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
		assertTest("append(Vector<int>*, int)", append(vec, 4)->toString() == res);
		assertTest("append(Matrix<int>*, Vector<int>*)", append(mat, vec)->toString() == matres);
	}

	void testConcatenate() {
		using namespace Cudheart::ArrayOps;
		using namespace Cudheart::MatrixOps;
		using namespace Cudheart::IO;
		string str = "1 2 3";
		string res = "[1, 2, 3, 1, 2, 3]";
		auto a = fromString<float>(str);
		auto b = fromString<float>(str);

		assertTest("concatenate(Vector<float>, Vector<float>)", concatenate(a, b)->toString() == res);

		string matres = "[\n [0, 1],\n [2, 3],\n [0, 1],\n [2, 3]\n]";

		auto c = arange(4, 2, 2);
		auto d = arange(4, 2, 2);

		assertTest("concatenate(Matrix<int>, Matrix<int>)", concatenate(c, d)->toString() == matres);
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
				assertTest("split(Vector<int>, int)", arr[i]->get(j) == a->get(i * (size / vecs) + j));
			}
		}

		auto b = (Matrix<int>*)a->reshape<int>(new Shape((int)(size / vecs), vecs));
		auto brr = split(b, size / vecs);

		for (int i = 0; i < size / vecs; i++) {
			for (int j = 0; j < vecs; j++) {
				assertTest("split(Matrix<int>, int)", brr[i]->get(j) == b->get(i * vecs + j));
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
				assertTest("tile(Vector<int>, int)", arr->get(i * size + j) == a->get(j));
			}
		}

		auto brr = tile(a, wReps, hReps);

		for (int i = 0; i < hReps; i++) {
			for (int j = 0; j < wReps; j++) {
				for (int k = 0; k < a->getSize(); k++) {
					assertTest("tile(Vector<int>, int, int)", brr->get(i, k + a->getSize() * j) == a->get(k));
				}
			}
		}

		auto c = (Matrix<int>*)(a->reshape<int>(new Shape(h, w)));
		auto crr = tile(c, reps);

		for (int i = 0; i < h; i++) {
			for (int j = 0; j < reps; j++) {
				for (int k = 0; k < w; k++) {
					assertTest("tile(Matrix<int>, int)", crr->get(i, k + w * j) == c->get(i, k));
				}
			}
		}

		auto drr = tile(c, wReps, hReps);

		for (int i = 0; i < hReps; i++) {
			for (int j = 0; j < h; j++) {
				for (int k = 0; k < wReps; k++) {
					for (int m = 0; m < w; m++) {
						assertTest("tile(Matrix<int>, int, int)", drr->get(j + i * h, m + k * w) == c->get(j, m));
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
				assertTest("remove(Vector<int>, int)", i == arr->get(idx));
				idx++;
			}

		}

		auto b = (Matrix<int>*)a->reshape<int>(new Shape(4, 5));
		auto brr = remove(b, rem);

		for (int i = 0; i < size - 1; i++) {
			assertTest("remove(Matrix<int>, int, axis=-1)", brr->get(i) == arr->get(i));
		}

		auto crr = (Matrix<int>*)(remove<int>(b, rem, 0));

		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 5; j++) {
				assertTest("remove(Matrix<int>, int, axis=0)", crr->get(i, j) == b->get(i, j));
			}
		}


		auto drr = (Matrix<int>*)(remove<int>(b, rem, 1));
	}

	void testTrimZeros() {
		using namespace Cudheart::VectorOps;
		using namespace Cudheart::ArrayOps;
		using namespace Cudheart::IO;

		auto a = fromString<int>("0, 0, 0, 1, 2, 3, 0, 2, 1, 0");

		assertTest("trimZeros(Vector<int>, 'fb')", trimZeros(a)->toString() == "[1, 2, 3, 0, 2, 1]");
		assertTest("trimZeros(Vector<int>, 'f')", trimZeros(a, "f")->toString() == "[1, 2, 3, 0, 2, 1, 0]");
		assertTest("trimZeros(Vector<int>, 'b')", trimZeros(a, "b")->toString() == "[0, 0, 0, 1, 2, 3, 0, 2, 1]");
	}

	void testUnique() {
		using namespace Cudheart::VectorOps;
		using namespace Cudheart::ArrayOps;
		using namespace Cudheart::IO;

		auto a = fromString<int>("0, 0, 0, 1, 3, 2, 0, 2, 1, 0");

		Vector<int>* res;


		res = unique(a, false, false, false)[0];
		assertTest("unique(a, false, false, false)", res->toString() == "[0, 1, 3, 2]");

		res = unique(a, true, false, false)[1];
		assertTest("unique(a, true, false, false)", res->toString() == "[0, 3, 4, 5]");

		res = unique(a, false, true, false)[2];
		assertTest("unique(a, false, true, false)", res->toString() == "[0, 0, 0, 1, 2, 3, 0, 3, 1, 0]");

		res = unique(a, false, false, true)[3];
		assertTest("unique(a, false, false, true)", res->toString() == "[5, 2, 1, 2]");
	}
}

namespace Cudheart::Testing::Arrays::IO {
	void test() {
		testFromString();
		testSave();
		testFromFile();
		testFromFunction();
	}

	void testFromString() {
		using namespace Cudheart::IO;
		string str = "1 2 3 4";
		string res = "[1, 2, 3, 4]";

		auto a = fromString<int>(str, ' ', 4);

		assertTest("fromString<int>(string, char, int)", a->toString() == res);

		auto b = fromString<int>(str);

		assertTest("fromString<int>(string)", b->toString() == res);

		auto c = fromString<int>(str, 3);

		assertTest("fromString<int>(string, int)", c->toString() == "[1, 2, 3]");
	}

	void testSave() {
		using namespace Cudheart::IO;
		string str = "11 21 31 41";
		string res = "[11, 21, 31, 41]";

		auto a = fromString<int>(str);
		save("savedArray.txt", a, 'x');

		string temp;
		string all;
		std::ifstream file("savedArray.txt");

		while (getline(file, temp)) {
			all += temp;
		}

		assertTest("save<int>(string, Vector<StringType*>*, char)", a->toString() == fromString<int>(all, 'x')->toString());

		file.close();

		auto b = fromString<int>(str);
		save("savedArray.txt", a);

		all = "";

		file = std::ifstream("savedArray.txt");

		while (getline(file, temp)) {
			all += temp;
		}

		assertTest("save<int>(string, Vector<StringType*>*)", a->toString() == fromString<int>(all)->toString());

		file.close();

		remove("savedArray.txt");
	}

	void testFromFile() {
		using namespace Cudheart::IO;
		string str = "11 21 31 41";
		string res = "[11, 21, 31, 41]";

		save("file.txt", fromString<int>(str));

		assertTest("fromFile<int>(string, char, int)", fromFile<int>("file.txt", ' ', 3)->toString() == "[11, 21, 31]");

		assertTest("fromFile<int>(string, char)", fromFile<int>("file.txt", ' ')->toString() == res);

		assertTest("fromFile<int>(string, int)", fromFile<int>("file.txt", 3)->toString() == "[11, 21, 31]");

		assertTest("fromFile<int>(string)", fromFile<int>("file.txt")->toString() == res);


		remove("file.txt");
	}

	int vectorFunction(int index) {
		return 10 * index;
	}

	void testFromFunction() {
		using namespace Cudheart::IO;
		auto vec = fromFunction(vectorFunction, 17);

		bool pass = true;
		for (int i = 0; i < vec->getSize(); i++) {
			pass = vec->get(i) == i * 10;
		}

		assertTest("fromFunction(int func(int), int)", pass);
	}
}

namespace Cudheart::Testing::Arrays::VecTest {
	void test() {
		testConstructors();
		testCastTo();
		testReshape();
	}

	void testConstructors() {
		int* arr = new int[] {5, 7, 451, 14, 25, 250, 52205, 255, 897};

		Vector<int>* vec = new Vector<int>(arr, 9);

		for (int i = 0; i < 9; i++) {
			assertTest("Vector<int>(int*, int)", vec->get(i) == arr[i]);
		}

		vec = new Vector<int>(7);
		for (int i = 0; i < 7; i++) {
			assertTest("Vector<int>(int)", vec->get(i) == vec->get(i));
		}

		vec = new Vector<int>({ 5, 7, 451, 14, 25, 250, 52205, 255, 897 });
		for (int i = 0; i < 9; i++) {
			assertTest("Vector<int>(initializer_list<int>)", vec->get(i) == arr[i]);
		}
	}

	void testCastTo() {
		int* arr = new int[] {5, 7, 451, 14, 25, 250, 52205, 255, 897};
		Vector<int>* a = new Vector<int>(arr, 9);

		Vector<StringType*>* b = a->castTo<StringType*>();
		
		assertTest("int to string cast", a->toString() == b->toString());

		Vector<ComplexType*>* c = a->castTo<ComplexType*>();
		Vector<ComplexType*>* d = b->castTo<ComplexType*>();

		assertTest("int and string to complex", c->toString() == d->toString());

		Vector<ComplexType*>* e = c->castTo<StringType*>()->castTo<ComplexType*>();

		assertTest("complex to string to complex", c->toString() == e->toString());

		Vector<float>* f = a->castTo<float>();

		for (int i = 0; i < f->getSize(); i++) {
			f->set(i, f->get(i) + 0.22);
		}

		assertTest("complex to string to complex", f->castTo<int>()->toString() == a->toString());
	}

	void testReshape() {

	}
}

