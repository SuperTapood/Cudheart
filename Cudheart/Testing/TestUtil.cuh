#pragma once

#include <string>
#include <chrono>
#include <ctime>
#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>

#include "../Cudheart/Cudheart.cuh"
#include "NumpyAPI/NumpyAPI.cuh"

using namespace std;

namespace Cudheart::Testing {
	class Testing {
	protected:
		string m_code;

		static Testing* self;

		static int tests;

		Testing();

	public:

		static Testing* get();

		static std::string exec(const char* cmd);

		static string procOutput(string output);

		static void submit(string name, string cmd, string output);

		static void testAll();
	};
}

