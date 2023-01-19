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
}