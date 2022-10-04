#pragma once

#include "Arrays/Arrays.cuh"
#include "Exceptions/Exceptions.cuh"
#include "Indexing/Indexing.cuh"
#include "Logic/Logic.cuh"
#include "Math/Math.cuh"
#include "Random/Random.cuh"
#include "StringTypes/StringTypes.cuh"
#include "Constants.cuh"


namespace Cudheart {
	std::string to_string(const StringType& c) {
		return c.m_str;
	}

	std::string to_string(const ComplexType& c) {
		string r = to_string(c.real);
		string i = to_string(c.imag);
		return r + " " + i + "i";
	}
}