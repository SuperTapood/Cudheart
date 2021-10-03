#include <cstdio>
#include <string>
#include <iostream>
#include <stdlib.h>
#include <stdexcept>

void raiseException(std::string cause) {
	std::cout << "Error: exception has been raised:\n" << cause << std::endl;
	throw std::runtime_error("cause");
}