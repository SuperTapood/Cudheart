#pragma once


#include <stdexcept>
#include <string>

using namespace std;


namespace Cudheart::Utils::Exceptions {
	class Exception {
	public:
		static void throwException(string message);
	};

	
	class BadShape : public Exception {
	public:
		static void raise(int req, int got);
	};
}


