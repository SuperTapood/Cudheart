#include <string>
#include <iostream>
#include <stdlib.h>

using namespace std;

void raiseException(string cause) {
	cout << "Error: exception has been raised:\n" << cause << std::endl;
	throw runtime_error("cause");
}