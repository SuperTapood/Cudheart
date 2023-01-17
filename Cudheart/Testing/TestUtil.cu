#pragma once

#include "TestUtil.cuh"

std::string exec(const char* cmd) {
	std::array<char, 128> buffer;
	std::string result;
	std::unique_ptr<FILE, decltype(&_pclose)> pipe(_popen(cmd, "r"), _pclose);
	if (!pipe) {
		throw std::runtime_error("popen() failed!");
	}
	while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
		result += buffer.data();
	}
	return result;
}

string procOutput(string output) {
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

void check(string name, string cmd, string output) {
	output = procOutput(output);

	// why
	ofstream file("file.py");
	cmd = "import numpy as np\n" + cmd + "\nprint(res.tolist(), end='')\n";
	file << cmd;
	file.close();

	/*cout << "test: " << name << endl;
	cout << exec("python file.py") << "END";*/
	string res = exec("python file.py");

	/*cout << "output: " << output << "END" << endl;
	cout << "res: " << res << "END" << endl;*/

	if (res != output) {
		cout << "Test " << name << " failed!\n";
		cout << "Cudheart generated: " << output;
		cout << "\nNumpy generated: " << res;
		cout << "\nCode Provided:\n\n" << cmd;
		exit(69);
	}
}

void check(string name, string output) {
	check(name, name, output);
}