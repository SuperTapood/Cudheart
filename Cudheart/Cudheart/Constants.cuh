#pragma once
#include <ctime>

namespace Cudheart {
	namespace Constants {
		static const double pi = 3.14159265358979323846;
		static const double euler = 2.71828182845904523536;
		static const double goldenRatio = 1.61803398874989484820;

		namespace {
			static size_t seed = 1;
			static bool isSet = false;
			static bool shouldThrow = false;
		}

		inline size_t getSeed() {
			if (!isSet) {
				return time(NULL);
			}
			return seed;
		}

		inline void setSeed(unsigned int newSeed) {
			isSet = true;
			seed = newSeed;
		}

		inline bool getShouldThrow() {
			return shouldThrow;
		}

		inline void setShouldThrow(bool should) {
			shouldThrow = should;
		}
	}
}