#pragma once

#include <fstream>
#include <iostream>
#include "../Arrays/NDArray.cuh"
#include "../Arrays/Vector.cuh"
#include "../Arrays/Matrix.cuh"
#include "../StringTypes/StringType.cuh"

using Cudheart::NDArrays::NDArray;
using Cudheart::NDArrays::Vector;
using Cudheart::NDArrays::Matrix;

namespace Cudheart {
	namespace IO {
		template <typename T>
		Vector<T>* fromString(std::string str, char sep, int count) {
			std::string temp = "";
			Vector<StringType*>* res = new Vector<StringType*>(count);
			int index = 0;
			for (int i = 0; i < str.size(); i++) {
				if (str.at(i) == sep) {
					res->set(index, new StringType(temp));
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

			if (index != count) {
				res->set(count - 1, new StringType(temp));
			}

			Vector<T>* out = res->castTo<T>();

			delete res;

			return out;
		}

		template <typename T>
		Vector<T>* fromString(std::string str, char sep) {
			int count = 1;
			for (int i = 0; i < str.size(); i++) {
				if (str.at(i) == sep) {
					count++;
				}
			}
			return fromString<T>(str, sep, count);
		}

		template <typename T>
		Vector<T>* fromString(std::string str, int count) {
			return fromString<T>(str, ' ', count);
		}

		template <typename T>
		Vector<T>* fromString(std::string str) {
			return fromString<T>(str, ' ');
		}

		template <typename T>
		Vector<T>* fromFile(std::string name, char sep, int count) {
			string temp;
			string all;
			std::ifstream file(name);

			while (getline(file, temp)) {
				all += temp;
			}

			Vector<T>* out = fromString<T>(all, sep, count);

			file.close();

			return out;
		}

		template <typename T>
		Matrix<T>* fromString(std::string str, char sep, int height, int width) {
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

			auto res = out->castTo<T>();

			delete out;

			return res;
		}

		template <typename T>
		Matrix<T>* fromString(std::string str, int height, int width) {
			return fromString<T>(str, ' ', height, width);
		}

		template <typename T>
		Vector<T>* fromFile(std::string name, char sep) {
			string temp;
			string all;
			std::ifstream file(name);

			while (getline(file, temp)) {
				all += temp;
			}

			Vector<T>* out = fromString<T>(all, sep);

			file.close();

			return out;
		}

		template <typename T>
		Vector<T>* fromFile(std::string name, int count) {
			return fromFile<T>(name, ' ', count);
		}

		template <typename T>
		Vector<T>* fromFile(std::string name) {
			return fromFile<T>(name, ' ');
		}

		template <typename T>
		Matrix<T>* fromFile(std::string name, char sep, int height, int width) {
			string temp;
			string all;
			std::ifstream file(name);

			while (getline(file, temp)) {
				all += temp;
			}

			Matrix<T>* out = fromString<T>(all, sep, height, width);

			file.close();

			return out;
		}

		template <typename T>
		Matrix<T>* fromFile(std::string name, int height, int width) {
			return fromFile<T>(name, ' ', height, width);
		}

		// redundent until i make a brand new binary file format
		/*template <typename T>
		Vector<T>* load(std::string name, char sep, int count) {
			Vector<StringType*>* vec = fromFile(name, sep, count);
			Vector<T>* out = new Vector<T>(count);
			for (int i = 0; i < count; i++) {
				out->set(i, (T)atof(vec->get(i)->c_str()));
			}
			return out;
		}
		template <typename T>
		Vector<T>* load(std::string name, char sep) {
			Vector<StringType*>* vec = fromFile(name, sep);
			Vector<T>* out = new Vector<T>(vec->getSize());
			for (int i = 0; i < vec->getSize(); i++) {
				out->set(i, (T)atof(vec->get(i)->c_str()));
			}
			return out;
		}
		template <typename T>
		Vector<T>* load(std::string name, int count) {
			Vector<StringType*>* vec = fromFile(name, count);
			Vector<T>* out = new Vector<T>(count);
			for (int i = 0; i < count; i++) {
				out->set(i, (T)atof(vec->get(i)->c_str()));
			}
			return out;
		}
		template <typename T>
		Vector<T>* load(std::string name) {
			Vector<StringType*>* vec = fromFile(name);
			int count = vec->getSize();
			Vector<T>* out = new Vector<T>(count);
			for (int i = 0; i < count; i++) {
				out->set(i, (T)atof(vec->get(i)->c_str()));
			}
			return out;
		}
		template <typename T>
		Matrix<T>* load(std::string name, char sep, int height, int width) {
			Matrix<StringType*>* mat = fromFile(name, sep, height, width);
			Matrix<T>* out = new Matrix<T>(height, width);
			for (int i = 0; i < out->getSize(); i++) {
				out->set(i, (T)atof(mat->get(i)->c_str()));
			}
			return out;
		}
		template <typename T>
		Matrix<T>* load(std::string name, int height, int width) {
			Matrix<StringType*>* mat = fromFile(name, ' ', height, width);
			Matrix<T>* out = new Matrix<T>(height, width);
			for (int i = 0; i < out->getSize(); i++) {
				out->set(i, (T)atof(mat->get(i)->c_str()));
			}
			return out;
		}*/

		template <typename T>
		void save(std::string name, NDArray<T>* arr, char sep) {
			std::ofstream file(name);
			for (int i = 0; i < arr->getSize() - 1; i++) {
				file << arr->getString(i) << sep;
			}
			file << arr->getString(-1);
			file.close();
		}

		template <typename T>
		void save(std::string name, NDArray<T>* arr) {
			save(name, arr, ' ');
		}

		template <typename T>
		Vector<T>* fromFunction(T(*func)(int), int size) {
			Vector<T>* out = new Vector<T>(size);
			for (int i = 0; i < size; i++) {
				out->set(i, func(i));
			}
			return out;
		}

		template <typename T>
		Matrix<T>* fromFunction(T(*func)(int, int), int height, int width) {
			Matrix<T>* out = new Matrix<T>(height, width);
			for (int i = 0; i < height; i++) {
				for (int j = 0; j < width; j++) {
					out->set(i, j, func(i, j));
				}
			}
			return out;
		}
	}
}