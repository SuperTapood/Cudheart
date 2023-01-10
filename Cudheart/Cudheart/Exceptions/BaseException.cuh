#pragma once

#include "../Util.cuh"

namespace Cudheart {
	namespace Exceptions {
		/// <summary>
		/// The base exception class for all of the other exceptions
		/// <para>
		/// Its aim is to provide a more friendly exception system,
		/// similar to that of higher languages
		/// </para>
		/// </summary>
		class BaseException : public std::exception {
		protected:
			/// <summary>
			/// the message to be presented in its fullest
			/// </summary>
			string m_msg;
		public:
			/// <summary>
			/// default constructor for easier inheritance
			/// </summary>
			BaseException();
			/// <summary>
			/// idk why this exists
			/// </summary>
			/// <param name="msg"> : string - the message to be presented</param>
			BaseException(string msg);
			/// <summary>
			/// raises the exception
			/// </summary>
			void raise();
			/// <summary>
			/// throw the exception
			/// </summary>
			void throwException();
			/// <summary>
			/// throw this exception
			/// </summary>
			/// <returns>the message to print</returns>
			const char* what() const throw ();
			/// <summary>
			/// print the exception
			/// </summary>
			void print();
		};
	}
}