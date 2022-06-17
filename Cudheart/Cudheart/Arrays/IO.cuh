#pragma once

#include <fstream>
#include <iostream>
#include "../Arrays/NDArray.cuh"
#include "../Arrays/Vector.cuh"
#include "../Arrays/Matrix.cuh"
#include "../StringOps/StringType.cuh"

using Cudheart::NDArrays::NDArray;
using Cudheart::NDArrays::Vector;
using Cudheart::NDArrays::Matrix;

namespace Cudheart::IO {
	inline Vector<Cudheart::StringType*>* fromString(std::string str, char sep, int count) {
		std::string temp = "";

		Vector<StringType*>* out = new Vector<StringType*>(count);
		int index = 0;
		for (int i = 0; i < str.size(); i++) {
			if (str.at(i) == sep) {
				out->set(index, new StringType(temp));
				temp = "";
				index++;
				if (index == count) {
					break;
				}
			}
			else {
				temp += str.at(i);
			}
		}

		return out;
	}

	inline Vector<Cudheart::StringType*>* fromString(std::string str, char sep) {
		int count = 0;
		for (int i = 0; i < str.size(); i++) {
			if (str.at(i) == sep) {
				count++;
			}
		}
		return fromString(str, sep, count);
	}

	inline Vector<Cudheart::StringType*>* fromString(std::string str, int count) {
		return fromString(str, ' ', count);
	}

	inline Vector<Cudheart::StringType*>* fromString(std::string str) {
		return fromString(str, ' ');
	}
	
	inline Vector<Cudheart::StringType*>* fromFile(std::string name, char sep, int count) {
		string temp;
		string all;
		std::ifstream file(name);

		while (getline(file, temp)) {
			all += temp;
		}
		
		Vector<Cudheart::StringType*>* out = fromString(all, sep, count);
		
		file.close();

		return out;
	}

	inline Vector<Cudheart::StringType*>* fromFile(std::string name, char sep) {
		string temp;
		string all;
		std::ifstream file(name);

		while (getline(file, temp)) {
			all += temp;
		}

		Vector<Cudheart::StringType*>* out = fromString(all, sep);

		file.close();

		return out;
	}

	inline Vector<Cudheart::StringType*>* fromFile(std::string name, int count) {
		return fromFile(name, ' ', count);
	}

	inline Vector<Cudheart::StringType*>* fromFile(std::string name) {
		return fromFile(name, ' ');
	}

	inline Matrix<Cudheart::StringType*>* fromString(std::string str, char sep, int height, int width) {
		std::string temp = "";

		Matrix<StringType*>* out = new Matrix<StringType*>(height, width);
		int count = height * width;
		int index = 0;
		for (int i = 0; i < str.size(); i++) {
			if (str.at(i) == sep) {
				out->set(index, new StringType(temp));
				temp = "";
				index++;
				if (index == count) {
					break;
				}
			}
			else {
				temp += str.at(i);
			}
		}

		return out;
	}

	inline Matrix<Cudheart::StringType*>* fromString(std::string str, int height, int width) {
		return fromString(str, ' ', height, width);
	}

	inline Matrix<Cudheart::StringType*>* fromFile(std::string name, char sep, int height, int width) {
		string temp;
		string all;
		std::ifstream file(name);
		int count = height * width;

		while (getline(file, temp)) {
			all += temp;
		}

		Matrix<Cudheart::StringType*>* out = fromString(all, sep, height, width);

		file.close();

		return out;
	}

	inline Matrix<Cudheart::StringType*>* fromFile(std::string name, int height, int width) {
		return fromFile(name, ' ', height, width);
	}

	template <typename T>
	Vector<T>* load(std::string name, char sep, int count) {
		return fromFile(name, sep, count)->castTo<T>();
	}

	template <typename T>
	Vector<T>* load(std::string name, char sep) {
		return fromFile(name, sep)->castTo<T>();
	}

	template <typename T>
	Vector<T>* load(std::string name, int count) {
		return fromFile(name, ' ', count)->castTo<T>();
	}

	template <typename T>
	Matrix<T>* load(std::string name, char sep, int height, int width) {
		return fromFile(name, sep, height, width)->castTo<T>();
	}
	
	template <typename T>
	Matrix<T>* load(std::string name, int height, int width) {
		return fromFile(name, ' ', height, width)->castTo<T>();
	}

	template <typename T>
	void save(std::string name, NDArray<T>* arr, char sep) {
		std::ofstream file(name);
		for (int i = 0; i < arr->getSize(); i++) {
			file << arr->get(i) << sep;
		}
		file.close();
	}

	template <typename T>
	void save(std::string name, NDArray<T>* arr) {
		save(name, arr, ' ');
	}
}