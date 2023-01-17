#include "NumpyAPI.cuh"

namespace Numpy {
	string createArray(string arr, string oName) {
		ostringstream os;
		os << oName << " = np.array(" << arr << ")\n";
		return os.str();
	}

	string createArray(string arr, string shape, string oName) {
		string out = createArray(arr, oName);
		out += oName + " = " + oName + ".reshape(" + shape + ")\n";
		return out;
	}

	string reshape(string name, string newShape, string oName) {
		ostringstream os;
		os << oName << " = " << name << ".reshape(" << newShape << ")\n";
		return os.str();
	}

	string empty(string shape, string oName) {
		ostringstream os;
		os << oName << " = np.empty(" << shape << ")\n";
		return os.str();
	}

	string T(string name, string oName) {
		ostringstream os;
		os << oName << " = " << name << ".T\n";
		return os.str();
	}

	string rot90(string name, int k, string oName) {
		return oName + " = np.rot90(" + name + ", k = " + to_string(k) + ")\n";
	}

	string augment(string a, string b, int axis, string oName) {
		return oName + " = np.concatenate((" + a + ", " + b + "), axis=" + to_string(axis) + ")\n";
	}
}