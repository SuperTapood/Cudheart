#include "TestUtil.cuh"

#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>
#include <iostream>
#include <vector>

namespace Cudheart::Testing {
	Testing* Testing::self = nullptr;
	int Testing::tests = 0;

	Testing::Testing() {
		m_code = "# this file is generated automatically to simplify unit testing :)\nfrom util import *\n\n";
	}

	Testing* Testing::get() {
		if (Testing::self == nullptr) {
			Testing::self = new Testing();
		}
		return Testing::self;
	}

	std::string Testing::exec(const char* cmd) {
		std::array<char, 128> buffer;
		std::string result;
		std::unique_ptr<FILE, decltype(&_pclose)> pipe(_popen(cmd, "rt"), _pclose);
		if (!pipe) {
			throw std::runtime_error("popen() failed!");
		}

		while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
			result += buffer.data();
		}
		return result;
	}

	string Testing::procOutput(string output) {
		auto arr = output.c_str();
		string res = "";

		for (int i = 0; i < output.size(); i++) {
			if (arr[i] == '\n') {
				if (arr[i - 1] != ',' && i + 1 < output.size() && arr[i + 1] == ' ') {
					i++;
				}
				continue;
			}
			res += arr[i];
		}

		/*cout << "input: " << output << endl;
		cout << "res: " << res << endl;*/
		return res;
	}

	void Testing::submit(string name, string cmd, string output) {
		output = procOutput(output);

		Testing* self = get();
		self->m_code += "test_name = \"" + name + "\"\n";
		self->m_code += cmd;
		if (cmd[cmd.size() - 1] != '\n') {
			self->m_code += "\n";
		}
		self->m_code += "out = " + output + "\n";
		self->m_code += "add2queue(test_name, res, out)\n\n\n";
		self->tests = self->tests + 1;

		// cout << "Test " + name + " submitted!\n";
	}

	void Testing::testAll() {
		get()->m_code += "print_res()\n";

		/*cout << "total code:\n\n";
		cout << get()->m_code;*/

		ofstream file("Testing\\python\\main.py");
		file << get()->m_code;
		file.close();

		string res = exec("python Testing\\python\\main.py");

		int tests = get()->tests;
		std::vector<string> names, np, cud, results;

		std::string delimiter = "|";

		size_t pos = 0;
		std::string token;
		int index = 0;
		string values[] = { "TEST_NAME", "CUDHEART RESULT", "NUMPY RESULT", "TEST RESULT" };

		while ((pos = res.find(delimiter)) != string::npos) {
			token = res.substr(0, pos);
			values[index++] = token;

			if (index == 4) {
				if (values[3] != "T") {
					cout << "Test " << values[0] << " failed!\n";
					cout << "Cudheart generated: " << values[2] << endl;
					cout << "Numpy generated:    " << values[1];
					// cout << "\nCode Provided:\n\n" << cmd;
					// cout << "\nPython Code:\n\n" << os.str();
					exit(69);
				}
				index = 0;
			}

			res.erase(0, pos + delimiter.length());
		}
	}
}