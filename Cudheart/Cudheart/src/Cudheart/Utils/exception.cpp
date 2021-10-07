#include <stdexcept>
#include <string>
#include <iostream>
#include <assert.h>

#include "exception.h"

using namespace std;

namespace Cudheart::Utils::Exceptions {
	void BadShape::raise(int req, int got) {
		string msg = "BadShape exception thrown; excpected length "
			+ to_string(req) + " got length " + to_string(got) + "\n";
		throwException(msg);
	}

	void ShapeHas0Dim::raise() {
		throwException("Got shape that contained dimension 0\n this is a headache i will not endure");
	}

	void Exception::throwException(string message)
	{
		// the best i can do right now, dont be mad
		printf(message.c_str());
		assert(false);
	}


}
