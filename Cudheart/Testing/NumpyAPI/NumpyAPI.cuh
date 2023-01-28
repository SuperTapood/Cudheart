#pragma once

#include <iostream>
#include <sstream>
#include <string>
#include <typeinfo>
#include <exception>
#include <climits>
#include <limits>
#include <stdio.h>
#include <stdarg.h>
#include <math.h>
#include <initializer_list>
#include <cstdarg>
#include <algorithm>
#include <array>
#include <iterator>
#include <vector>
#include <string>

using namespace std;

namespace Numpy {
	string createArray(string arr, string oName);

	string createArray(string arr, string shape, string oName);

	string reshape(string name, string newShape, string oName);

	string empty(string shape, string oName);

	string full(string shape, string value, string oName);

	string T(string name, string oName);

	string rot90(string name, int k, string oName);

	string augment(string a, string b, int axis, string oName);

	string fromFile(string file, char sep, string dtype, string oName);

	string fromFunction(string funcName, string shape, string dtype, string oName);

	string arange(string start, string stop, string step, string dtype, string oName);

	string linspace(string start, string stop, string num, string endpoint, string dtype, string oName);

	string logspace(string start, string stop, string num, string endpoint, string base, string dtype, string oName);

	string geomspace(string start, string stop, string num, string endpoint, string dtype, string oName);

	string eye(string N, string M, string k, string dtype, string oName);

	string meshgrid(string a, string b, string oName);

	string append(string a, string b, string axis, string oName);

	string concatenate(string a, string b, string oName);

	string diag(string v, string k, string oName);

	string diagflat(string v, string k, string oName);

	string split(string ary, string nums, string oName);

	string tile(string a, string reps, string oName);

	string remove(string a, string index, string oName);

	string remove(string a, string index, string axis, string oName);

	string trim_zeros(string filt, string trim, string oName);

	string unique(string ar, string oName);

	string tri(string N, string M, string k, string dtype, string oName);

	string tril(string M, string k, string oName);

	string triu(string M, string k, string oName);

	string vander(string x, string N, string increasing, string oName);

	string argmax(string a, string oName);

	string argmin(string a, string oName);

	string nonzero(string a, string oName);

	string argwhere(string a, string oName);

	string flatnonzero(string a, string oName);

	string where(string condition, string x, string y, string oName);

	string searchsorted(string a, string v, string side, string sorter, string oName);

	string extract(string condition, string arr, string oName);

	string count_nonzero(string a, string oName);

	string sort(string a, string kind, string oName);

	string argsort(string a, string kind, string oName);

	string partition(string a, string kth, string oName);

	string argpartition(string a, string kth, string oName);

	string all(string arr, string oName);

	string any(string arr, string oName);

	string logicalAnd(string a, string b, string oName);

	string logicalOr(string a, string b, string oName);

	string logicalNot(string arr, string oName);

	string logicalXor(string a, string b, string oName);

	string isclose(string a, string b, string rtol, string atol, string oName);

	string allclose(string a, string b, string rtol, string atol, string oName);

	string equals(string a, string b, string oName);

	string greater(string a, string b, string oName);

	string greaterEquals(string a, string b, string oName);

	string less(string a, string b, string oName);

	string lessEqual(string a, string b, string oName);

	string maximum(string a, string b, string oName);

	string amax(string x, string oName);

	string minimum(string a, string b, string oName);

	string amin(string x, string oName);
}