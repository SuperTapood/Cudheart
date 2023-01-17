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

	string full(string shape, string value, string oName) {
		return oName + " = np.full(" + shape + ", " + value + ")";
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

	string fromFile(string file, char sep, string dtype, string oName) {
		return oName + " = np.fromfile('" + file + "', sep='" + sep + "', dtype=" + dtype + ")\n";
	}

	string fromFunction(string funcName, string shape, string dtype, string oName) {
		return oName + " = np.fromfunction(" + funcName + ", " + shape + ", dtype=" + dtype + ")\n";
	}

	string arange(string start, string stop, string step, string dtype, string oName) {
		return oName + " = np.arange(" + start + ", " + stop + ", " + step + ", dtype=" + dtype + ")";
	}

	string linspace(string start, string stop, string num, string endpoint, string dtype, string oName) {
		return oName + " = np.linspace(" + start + ", " + stop + ", " + num + ", " + endpoint + ", dtype=" + dtype + ")";
	}

	string logspace(string start, string stop, string num, string endpoint, string base, string dtype, string oName) {
		return oName + " = np.logspace(" + start + ", " + stop + ", " + num + ", " + endpoint + ", dtype=" + dtype + ", base=" + base + ")";
	}

	string geomspace(string start, string stop, string num, string endpoint, string dtype, string oName) {
		return oName + " = np.geomspace(" + start + ", " + stop + ", " + num + ", " + endpoint + ", dtype=" + dtype + ")";
	}
}