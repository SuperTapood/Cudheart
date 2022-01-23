#pragma once

#include "../Inc.h"

class BaseException : public exception {
protected:
	string msg;
public:
	BaseException();
	BaseException(string msg);
	const char* what() const throw();
	void print();
};