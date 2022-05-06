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

	void BaseException::raise()
	{
		cout << endl << m_msg << endl;
		exit(1);
	}

	void BaseException::throwException()
	{
		throw BaseException(m_msg);
	}

	const char* BaseException::what() const throw ()
	{
		return m_msg.c_str();
	}

	void BaseException::print() {
		cout << m_msg << endl;
	}
}