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


void check(string name, string cmd, string output) {
	// why
	ofstream file("file.py");
	file << "import numpy as np\n";
	file << cmd;
	file << "\nprint(res.tolist(), end='')";
	file.close();

	/*cout << "test: " << name << endl;
	cout << exec("python file.py") << "END";*/
	string res = exec("python file.py");

	/*cout << "output: " << output << "END" << endl;
	cout << "res: " << res << "END" << endl;*/

	if (res != output) {
		cout << "Test " << name << " failed!\n";
		exit(69);
	}
}

void check(string name, string output) {
	check(name, name, output);
}