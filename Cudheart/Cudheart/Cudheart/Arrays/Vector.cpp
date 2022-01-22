#include "Vector.h"

Vector::Vector(int size) {
	this->size = size;
}

//std::ostream& Vector::operator<<(std::ostream& out)
//{
//	Vector vec = *this;
//	out << "[";
//	for (unsigned int j = 0; j < vec.size; j++)
//	{
//		if (j % vec.size == vec.size - 1)
//			out << vec[j] << "]" << std::endl;
//		else
//			out << vec[j] << ", ";
//	}
//	return out;
//}

//int Vector::operator[](std::size_t index)
//{
//	return arr[index];
//}


//string Vector::toString() {
//	Vector vec = *this;
//	ostringstream out;
//	out << "[";
//	for (unsigned int j = 0; j < vec.size; j++)
//	{
//		if (j % vec.size == vec.size - 1)
//			out << vec[j] << "]" << std::endl;
//		else
//			out << vec[j] << ", ";
//	}
//	return out.str();
//}