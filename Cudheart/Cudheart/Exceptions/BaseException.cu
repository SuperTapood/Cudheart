#include "BaseException.cuh"


BaseException::BaseException()
{
	this->msg = "";
}

BaseException::BaseException(string msg)
{
	this->msg = msg;
}

const char* BaseException::what() const throw()
{
	return msg.c_str();
}

void BaseException::print() {
	cout << msg << endl;
}