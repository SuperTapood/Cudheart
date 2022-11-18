#include "TestArray.cuh"

void testArray() {
	testArrOps();
	//testIO();
}

#pragma region ArrOpsTesting

void testArrOps() {
	testAppend();
}

void testAppend() {
	using namespace Cudheart::ArrayOps;
	using namespace Cudheart::MatrixOps;
	using namespace Cudheart::IO;
	string str = "1 2 3";
	string res = "[1, 2, 3, 4]";
	string matres ="[\n [0, 1, 2],\n [3, 4, 5],\n [6, 7, 8],\n [1, 2, 3]\n]";
	string filename = "file.txt";
	save("file.txt", fromString(str));
	auto vec = load<int>(filename);
	auto mat = arange(9, 3, 3);
	assertTest(ARRAYS_ARROPS_APPEND_VecT, append(vec, 4)->toString() == res);
	assertTest(ARRAYS_ARROPS_APPEND_VecT, append(mat, vec)->toString() == matres);
}

#pragma endregion

#pragma region IOTesting
void testIO() {
	testFromString();
	testSave();
	testFromFile();
	testLoad();
	testFromFunction();
}

void testFromString() {
	using namespace Cudheart::IO;
	string str = "1 2 3 4";
	string res = "[1, 2, 3, 4]";

	auto a = fromString(str, ' ', 4);

	assertTest(ARRAYS_IO_FROM_STRING_SCI, a->toString() == res);

	auto b = fromString(str);

	assertTest(ARRAYS_IO_FROM_STRING_S, b->toString() == res);

	auto c = fromString(str, 3);

	assertTest(ARRAYS_IO_FROM_STRING_SI, c->toString() == "[1, 2, 3]");
}

void testSave() {
	using namespace Cudheart::IO;
	string str = "11 21 31 41";
	string res = "[11, 21, 31, 41]";

	auto a = fromString(str);
	save("savedArray.txt", a, 'x');

	string temp;
	string all;
	std::ifstream file("savedArray.txt");

	while (getline(file, temp)) {
		all += temp;
	}

	assertTest(ARRAYS_IO_SAVE_SAC, a->toString() == fromString(all, 'x')->toString());

	file.close();

	auto b = fromString(str);
	save("savedArray.txt", a);

	all = "";

	file = std::ifstream("savedArray.txt");

	while (getline(file, temp)) {
		all += temp;
	}

	assertTest(ARRAYS_IO_SAVE_SA, a->toString() == fromString(all)->toString());

	file.close();

	remove("savedArray.txt");
}

void testFromFile() {
	using namespace Cudheart::IO;
	string str = "11 21 31 41";
	string res = "[11, 21, 31, 41]";

	save("file.txt", fromString(str));

	assertTest(ARRAYS_IO_FROM_FILE_SCI, fromFile("file.txt", ' ', 3)->toString() == "[11, 21, 31]");

	assertTest(ARRAYS_IO_FROM_FILE_SC, fromFile("file.txt", ' ')->toString() == res);

	assertTest(ARRAYS_IO_FROM_FILE_SI, fromFile("file.txt", 3)->toString() == "[11, 21, 31]");

	assertTest(ARRAYS_IO_FROM_FILE_S, fromFile("file.txt")->toString() == res);


	remove("file.txt");
}

void testLoad() {
	using namespace Cudheart::IO;
	using namespace Cudheart::VectorOps;

	string str = "1 2 3 4";
	string res = "[1, 2, 3, 4]";

	save("file.txt", fromString(str));

	assertTest(ARRAYS_IO_LOAD_SCI, load<float>("file.txt", ' ', 4)->toString() == "[1, 2, 3, 4]");

	assertTest(ARRAYS_IO_LOAD_SC, load<int>("file.txt", ' ')->toString() == res);

	assertTest(ARRAYS_IO_LOAD_SI, load<int>("file.txt", 3)->toString() == "[1, 2, 3]");

	assertTest(ARRAYS_IO_LOAD_S, load<int>("file.txt")->toString() == res);


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

	assertTest(ARRAYS_IO_FROM_FUNCTION_FuncI, pass);
}
#pragma endregion
