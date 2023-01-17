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
	ostringstream os;
	os << "import numpy as np\n";
	os << cmd << "\n";
	// os << "shap = res.shape if type(res) == np.ndarray else None\n";
	// os << "res = [round(i, 18) for i in res.flatten().tolist()] if type(res) == np.ndarray else res\n";
	// os << "res = np.array(res).reshape(shap).tolist() if shap is not None else res\n";
	os << "close = np.allclose(res, np.array(" << output << "))\n";
	os << "print(str((res.tolist() if type(res) == np.ndarray else res)) + ('T' if close else 'F'), end = '')\n";
	file << os.str();
	file.close();

	/*cout << "test: " << name << endl;
	cout << exec("python file.py") << "END";*/
	string res = exec("python file.py");
	bool pass = false;


	if (res.size() != 0) {
		pass = res[res.size() - 1] == 'T';
		res.pop_back();
	}

	/*cout << "output: " << output << "END" << endl;
	cout << "res: " << res << "END" << endl;*/

	if (!pass) {
		cout << "Test " << name << " failed!\n";
		cout << "Cudheart generated: " << output;
		cout << "\nNumpy generated:    " << res;
		// cout << "\nCode Provided:\n\n" << cmd;
		// cout << "\nPython Code:\n\n" << os.str();
		exit(69);
	}
}

void check(string name, string output) {
	check(name, name, output);
}