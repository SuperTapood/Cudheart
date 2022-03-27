#pragma once

#include "../Util.cuh"

/// <summary>
/// The base exception class for all of the other exceptions
/// <para>
/// Its aim is to provide a more friendly exception system,
/// similar to that of higher languages
/// </para>
/// </summary>
class BaseException : public exception {
protected:
	/// <summary>
	/// the message to be presented in its fullest
	/// </summary>
	string msg;
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
	/// i have no idea what that does actually
	/// </summary>
	/// <returns> c string - the presented message</returns>
	const char* what() const throw();
	/// <summary>
	/// print the exception
	/// </summary>
	void print();
};