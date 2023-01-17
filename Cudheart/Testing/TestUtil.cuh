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

