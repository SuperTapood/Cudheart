#include "BaseException.cuh"
#include "../Constants.cuh"

namespace Cudheart {
	namespace Exceptions {
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
			if (Cudheart::Constants::getShouldThrow()) {
				throwException();
			}
			cout << m_msg << endl;
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
}