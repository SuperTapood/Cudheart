#pragma once

#include "pch.h"

#if defined _DEBUG
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

	using std::exception;
	using std::cout;
	using std::cin;
	using std::endl;
	using std::string;
	using std::ostringstream;
	using std::ostream;
	using std::initializer_list;
	using std::to_string;
#endif