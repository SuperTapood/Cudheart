#include "TestArray.cuh"

void testArray() {
	testIO();
}

void testIO() {
	testFromString();
	testSave();
	testFromFile();
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
}