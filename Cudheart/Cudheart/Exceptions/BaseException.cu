#include "BaseException.cuh"


namespace Cudheart::Exceptions {
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
}