#include "TestArray.cuh" 

void testArray() {
	auto start = std::chrono::system_clock::now();
	testArrOps();
	testIO();
	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed = end - start;
	// cout << "array tests passed in " << elapsed.count() << "s\n";	
}

#pragma region ArrOpsTesting

void testArrOps() {
	testAppend();
	testConcatenate();
	testSplit();
	testTile();
	testRemove();
}

void testAppend() {
	using namespace Cudheart::ArrayOps;
	using namespace Cudheart::MatrixOps;
	using namespace Cudheart::IO;
	string str = "1 2 3";
	string res = "[1, 2, 3, 4]";
	string matres ="[\n [0, 1, 2],\n [3, 4, 5],\n [6, 7, 8],\n [1, 2, 3]\n]";
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

	auto b = (Matrix<int>*)a->shapeLike<int>(new Shape((int)(size / vecs), vecs));
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

	auto c = (Matrix<int>*)(a->shapeLike<int>(new Shape(h, w)));
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

	auto b = (Matrix<int>*)a->shapeLike<int>(new Shape(4, 5));
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


#pragma endregion

#pragma region IOTesting
void testIO() {
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
#pragma endregion
