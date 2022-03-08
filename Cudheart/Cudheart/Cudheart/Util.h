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
#include <string>

using std::exception;
using std::cout;
using std::cin;
using std::endl;
using std::string;
using std::ostringstream;
using std::ostream;
using std::initializer_list;
using std::to_string;
using std::array;
using std::copy;
using std::begin;
using std::end;


#ifndef print
	void print(auto msg) {
		cout << msg << endl;
	}
#else
	#undef print
	void print(string msg) {
		cout << msg << endl;
	}
#endif