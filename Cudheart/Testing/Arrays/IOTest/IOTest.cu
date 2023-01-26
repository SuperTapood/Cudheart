#include "IOTest.cuh"
#include <string>

using namespace std;

namespace Cudheart::Testing::Arrays::IO {
	void test() {
		testFromString();
		// todo figure out how to get funny c++ to get python to write files
		// testSave();
		testFromFile();
		testFromFunction();
	}

	void testFromString() {
		using namespace Cudheart::IO;
		string str = "1 2 3 4";
		string res = "[1, 2, 3, 4]";
		string cmd;

		auto a = fromString<int>(str, ' ', 4);

		cmd = Numpy::createArray(res, "res");

		Testing::submit("IO::fromString<int>(string, char, int)", cmd, a->toString());

		auto b = fromString<int>(str);

		cmd = Numpy::createArray(res, "res");

		Testing::submit("IO::fromString<int>(string)", cmd, b->toString());

		auto c = fromString<int>(str, 3);

		cmd = Numpy::createArray("[1, 2, 3]", "res");

		Testing::submit("IO::fromString<int>(string, int)", cmd, c->toString());
	}

	void testSave() {
		using namespace Cudheart::IO;
		string str = "11 21 31 41";
		string res = "[11, 21, 31, 41]";
		string cmd;

		auto a = fromString<int>(str);
		save("Testing\\python\\savedArray.txt", a, 'x');

		string temp;
		string all;
		std::ifstream file("savedArray.txt");

		while (getline(file, temp)) {
			all += temp;
		}

		cmd = Numpy::fromFile("savedArray.txt", 'x', "int", "res");

		Testing::submit("IO::save<int>(string, Vector<StringType*>*, char)", cmd, a->toString());

		file.close();

		auto b = fromString<int>(str);
		save("savedArray.txt", a);

		all = "";

		file = std::ifstream("savedArray.txt");

		while (getline(file, temp)) {
			all += temp;
		}

		cmd = Numpy::fromFile("savedArray.txt", ' ', "int", "res");

		Testing::submit("IO::save<int>(string, Vector<StringType*>*)", cmd, a->toString());

		file.close();

		//remove("savedArray.txt");
	}

	void testFromFile() {
		using namespace Cudheart::IO;
		string str = "11 21 31 41";
		string res = "[11, 21, 31, 41]";
		string cmd;

		save("Testing\\python\\file.txt", fromString<int>(str));

		cmd = Numpy::fromFile("file.txt", ' ', "int", "res");
		cmd += "res = res[0:3]";

		Testing::submit("IO::fromFile<int>(string, char, int)", cmd, fromFile<int>("file.txt", ' ', 3)->toString());

		cmd = Numpy::fromFile("file.txt", ' ', "int", "res");

		Testing::submit("IO::fromFile<int>(string, char)", cmd, fromFile<int>("file.txt", ' ')->toString());

		cmd = Numpy::fromFile("file.txt", ' ', "int", "res");
		cmd += "res = res[0:3]";

		Testing::submit("IO::fromFile<int>(string, int)", cmd, fromFile<int>("file.txt", 3)->toString());

		cmd = Numpy::fromFile("file.txt", ' ', "int", "res");

		Testing::submit("IO::fromFile<int>(string)", cmd, fromFile<int>("file.txt")->toString());

		//remove("file.txt");
	}

	int vectorFunction(int index) {
		return 10 * index;
	}

	void testFromFunction() {
		using namespace Cudheart::IO;
		auto vec = fromFunction(vectorFunction, 17);
		string cmd;

		bool pass = true;
		for (int i = 0; i < vec->getSize(); i++) {
			pass = vec->get(i) == i * 10;
		}

		cmd = "func = lambda x: 10 * x\n";
		cmd += Numpy::fromFunction("func", "(17,)", "int", "res");

		Testing::submit("IO::fromFunction(int func(int), int)", cmd, vec->toString());
	}
}