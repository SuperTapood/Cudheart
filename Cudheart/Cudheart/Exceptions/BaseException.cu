#include "BaseException.cuh"


namespace Cudheart::Exceptions {
	BaseException::BaseException()
	{
		m_msg = "";
	}

	BaseException::BaseException(string msg)
	{
		m_msg = msg;
	}

	const char* BaseException::what() const throw()
	{
		return m_msg.c_str();
	}

	void BaseException::print() {
		cout << m_msg << endl;
	}
}