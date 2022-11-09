#pragma once
#include "../../Arrays/Arrays.cuh"
#include <stdio.h>
#include "../TestUtil.cuh"

void testArray();

#pragma region ArrOpsTesting
void testArrOps();

void testAppend();
#pragma endregion

#pragma region IOTesting
void testIO();

void testFromString();

void testSave();

void testFromFile();

void testLoad();

void testFromFunction();
#pragma endregion