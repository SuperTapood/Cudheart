#include "IOTest.cuh"
#include <string>

using namespace std;

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

		check("IO::fromString<int>(string, char, int)", a->toString());

		auto b = fromString<int>(str);

		check("IO::fromString<int>(string)", b->toString());

		auto c = fromString<int>(str, 3);

		check("IO::fromString<int>(string, int)", c->toString());
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

		check("IO::save<int>(string, Vector<StringType*>*, char)", a->toString());

		file.close();

		auto b = fromString<int>(str);
		save("savedArray.txt", a);

		all = "";

		file = std::ifstream("savedArray.txt");

		while (getline(file, temp)) {
			all += temp;
		}

		check("IO::save<int>(string, Vector<StringType*>*)", a->toString());

		file.close();

		remove("savedArray.txt");
	}

	void testFromFile() {
		using namespace Cudheart::IO;
		string str = "11 21 31 41";
		string res = "[11, 21, 31, 41]";

		save("file.txt", fromString<int>(str));

		check("IO::fromFile<int>(string, char, int)", fromFile<int>("file.txt", ' ', 3)->toString());

		check("IO::fromFile<int>(string, char)", fromFile<int>("file.txt", ' ')->toString());

		check("IO::fromFile<int>(string, int)", fromFile<int>("file.txt", 3)->toString());

		check("IO::fromFile<int>(string)", fromFile<int>("file.txt")->toString());

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

		check("IO::fromFunction(int func(int), int)", vec->toString());
	}
}