#pragma once


#include <stdexcept>
#include <string>

using namespace std;


namespace Cudheart::Utils::Exceptions {
	class Exception {
	public:
		Exception(char const* const message) throw();
		Exception(string message) throw();
		string toString();
		const char* toCString();
	public:
		const char* message;
	};
}


