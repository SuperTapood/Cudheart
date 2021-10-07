#pragma once


#include "../Core/Shape.h"
#include "../Dtypes/Dtype.h"



namespace Cudheart::CObjects {
	template <class T>
	class CVector {
	public:
		// general constructor.
		// i would add more, but fuck me that would be pain
		// @param T arr[] - the array as a void pointer
		// @param Shape shape - the shape of the vector (should be of length 1)
		// @param Dtype dytpe - the dtype to be casted on the vector
		// @param bool copy - whether or not to copy the array
		// @returns the object itself ya big dum
		CVector(T arr[], Shape shape, Dtype dtype, bool copy);

		// constructor that assumes the object should be copied.
		// @param T arr[] - the array as a void pointer
		// @param Shape shape - the shape of the vector (should be of length 1)
		// @param Dtype dytpe - the dtype to be casted on the vector
		// @returns the object itself ya big dum 
		CVector(T arr[], Shape shape, Dtype dtype);
		// destroy the CVector object
		~CVector();
	private:
		// initialiazes the object. exists to that i may overload constructors
		// @param void* arr - the array as a void pointer
		// @param Shape shape - the shape of the vector (should be of length 1)
		// @param Dtype dytpe - the dtype to be casted on the vector
		// @param bool copy - whether or not to copy the array
		void init(T arr[], Shape shape, Dtype dtype, bool copy);
	public:
		// pointer to the array.
		// void because i do not discriminate, and to allow user types along the line
		// whenever the array needs to be accssed, it will be casted using the dtype
		// techincally a doing dtype.cast(&arr) should allow
		// dtype to cast void* into an array of any type
		T* ptr;
		Shape shape;
		Dtype dtype;
	};
}