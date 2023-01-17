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

	string T(string name, string oName);

	string rot90(string name, int k, string oName);

	string augment(string a, string b, int axis, string oName);
}