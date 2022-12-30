#pragma once
#include "TestUtil.cuh"
#include <iostream>

void check(string id, bool b) {
	if (b) {
		// cout << "Test " << id << " passed!\n";
		return;
	}
	cout << "Tests Failed! " << id << " failed assertion\n";
	exit(1);
}