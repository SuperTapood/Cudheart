#pragma once
#include <stdio.h>
#include "../TestUtil.cuh"

void testArray();

#pragma region ArrOpsTesting
void testArrOps();

void testAppend();

void testConcatenate();

void testSplit();

void testTile();

void testRemove();

#pragma endregion

#pragma region IOTesting
void testIO();

void testFromString();

void testSave();

void testFromFile();

void testFromFunction();
#pragma endregion