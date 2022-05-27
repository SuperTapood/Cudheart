#pragma once

namespace Cudheart::Constants {
	static const double pi = 3.14159265358979323846;
	static const double euler = 2.71828182845904523536;
	static const double goldenRatio = 1.61803398874989484820;
	namespace {
		static unsigned int seed = 1;
		static bool isSet = false;
	}
	
	inline unsigned int getSeed() {
		if (!isSet) {
			return time(NULL);
		}
		return seed;
	}

	inline void setSeed(unsigned int newSeed) {
		isSet = true;
		Constants::seed = newSeed;
	}
}