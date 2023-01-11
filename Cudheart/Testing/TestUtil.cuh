#pragma once

#include <string>
#include <chrono>
#include <ctime>

#include "../Cudheart/Cudheart.cuh"

using namespace std;

template <typename T, typename U>
void check(string id, T a, U b) {
	if (a == b) {
		// cout << "Test " << id << " passed!\n";
		return;
	}
	cout << "Tests Failed! " << id << " failed assertion (" << a << " does not equal " << b << ")\n";
	exit(1);
}

template <typename T>
void check(string id, T a) {
	check(id, a, true);
}