#pragma once

#include <string>

using namespace std;

#include "Test/test.h"
#include "Core/core.h"
#include "CLib/CLib.h"

namespace Cudheart {
	int MAJOR_VERSION = 1;
	int MINOR_VERSION = 0;
	int PATCH_VERSION = 0;
	string version = to_string(MAJOR_VERSION) + "."
		+ to_string(MINOR_VERSION) + "."
		+ to_string(PATCH_VERSION);

	Math math;
	Random rand;
	Creation create;
	
	void init() {
		// currently not using cuda!
		Lib lib = CLib();
		math = lib.math;
		rand = lib.rand;
		create = lib.create;
	}
}