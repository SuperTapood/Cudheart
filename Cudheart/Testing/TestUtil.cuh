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

std::string exec(const char* cmd);

void check(string cmd, string output);

void check(string name, string cmd, string output);

//template <typename T, typename U>
//void check(string id, T a, U b) {
//	if (a == b) {
//		//cout << "Test " << id << " passed!\n";
//		return;
//	}
//	cout << "Tests Failed! " << id << " failed assertion (" << a << " does not equal " << b << ")\n";
//	exit(1);
//}
//
//template <typename T>
//void check(string id, T a) {
//	check(id, a, true);
//}

