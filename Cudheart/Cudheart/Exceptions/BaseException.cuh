#pragma once

#include "../Util.cuh"

namespace Cudheart::Exceptions {
	/// <summary>
	/// The base exception class for all of the other exceptions
	/// <para>
	/// Its aim is to provide a more friendly exception system,
	/// similar to that of higher languages
	/// </para>
	/// </summary>
	class BaseException {
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
		/// print the exception
		/// </summary>
		void print();
	};
}