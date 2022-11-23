#include "TestArray.cuh"

void testArray() {
	testArrOps();
	testIO();
}

#pragma region ArrOpsTesting

void testArrOps() {
	testAppend();
	testConcatenate();
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
	assertTest("append<int>(Vector<int>*, int)", append(vec, 4)->toString() == res);
	assertTest("append<int>(Matrix<int>*, Vector<int>*)", append(mat, vec)->toString() == matres);
}

void testConcatenate() {
	using namespace Cudheart::ArrayOps;
	using namespace Cudheart::MatrixOps;
	using namespace Cudheart::IO;
	string str = "1 2 3";
	string res = "[1, 2, 3, 1, 2, 3]";
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
