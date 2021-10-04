#include <stdexcept>
#include <string>
#include <iostream>

#include "exception.h"

using namespace std;

namespace Cudheart::Utils::Exceptions {
	Exception::Exception(string message) {
		this->message = message.c_str();
	}

	Exception::Exception(char const* const message) {
		this->message = message;
	}


	string Exception::toString() {
		return string(message);
	}

	const char* Exception::toCString() {
		return message;
	}
}
