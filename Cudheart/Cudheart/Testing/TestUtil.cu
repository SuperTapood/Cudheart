#pragma once
#include "TestUtil.cuh"
#include <iostream>

void assertTest(string id, bool b) {
	if (b) {
		// cout << "Test ID " << id << " passed!\n";
		return;
	}
	cout << "Tests Failed! Test (ID " << id << ") failed assertion\n";
	exit(1);
}