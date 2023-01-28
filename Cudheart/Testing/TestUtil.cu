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
		m_code = "# this file is generated automatically to simplify unit testing :)\n\nfrom util import *\n\nimport warnings\n\nwarnings.filterwarnings('ignore', category = DeprecationWarning)\n\n";
		m_code += "# anti future headache tech\ntrue = True\nfalse = False\n\n";
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
		self->m_code += cmd + "\n";
		self->m_code += "out = " + output + "\n";
		self->m_code += "add2queue(test_name, res, out)\n\n\n\n";
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
		int indices[] = { 0, 0, 0, 0 };
		while ((pos = res.find(delimiter)) != std::string::npos) {
			token = res.substr(0, pos);
			switch (index % 4) {
			case 0:
				names.push_back(token);
				break;
			case 1:
				np.push_back(token);
				break;
			case 2:
				cud.push_back(token);
				break;
			case 3:
				results.push_back(token);
				break;
			}
			index++;
			res.erase(0, pos + delimiter.length());
		}

		for (int i = 0; i < tests; i++) {
			if (results[i] != "T") {
				cout << "Test " << names[i] << " failed!\n";
				cout << "Cudheart generated: " << cud[i] << endl;
				cout << "Numpy generated:    " << np[i];
				// cout << "\nCode Provided:\n\n" << cmd;
				// cout << "\nPython Code:\n\n" << os.str();
				exit(69);
			}

			// cout << "Test " + results[i] + "passed!\n";
		}
	}
}