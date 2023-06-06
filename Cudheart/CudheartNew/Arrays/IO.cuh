#pragma once

#include "NDArray.cuh"
#include <fstream>
#include <functional>

namespace CudheartNew::IO {
	template <typename T>
	NDArray<T>* fromString(std::string& str, char sep, int count) {
		std::string temp = "";
		auto out = new NDArray<T>({ count });
		int index = 0;

		for (auto c : str) {
			if (c == sep) {
				if (index == count) {
					break;
				}
				out->at(index++) = temp;
				temp = "";
			}
			else {
				temp += c;
			}
		}

		if (index != count) {
			out->at(count - 1) = temp;
		}

		return out;
	}

	template <typename T>
	NDArray<T>* fromString(std::string& str, char sep) {
		int count = 1;
		for (int i = 0; i < str.size(); i++) {
			if (str.at(i) == sep) {
				count++;
			}
		}
		return fromString<T>(str, sep, count);
	}

	template <typename T>
	NDArray<T>* fromString(std::string& str, int count) {
		return fromString<T>(str, ' ', count);
	}

	template <typename T>
	NDArray<T>* fromString(std::string& str) {
		return fromString<T>(str, ' ');
	}

	template <typename T>
	NDArray<T>* fromFile(std::string name, char sep, int count) {
		std::string temp;
		std::string all;
		std::ifstream file(name);

		while (std::getline(file, temp)) {
			all += temp;
		}

		NDArray<T>* out = fromString<T>(all, sep, count);

		file.close();

		return out;
	}

	template <typename T>
	NDArray<T>* fromFile(std::string name, char sep) {
		std::string temp;
		std::string all;
		std::ifstream file(name);

		while (std::getline(file, temp)) {
			all += temp;
		}

		NDArray<T>* out = fromString<T>(all, sep);

		file.close();

		return out;
	}

	template <typename T>
	NDArray<T>* fromFile(std::string name, int count) {
		return fromFile<T>(name, ' ', count);
	}

	template <typename T>
	NDArray<T>* fromFile(std::string name) {
		return fromFile<T>(name, ' ');
	}

	template <typename T>
	void save(std::string name, NDArray<T>* arr, char sep) {
		std::ofstream file(name);
		for (int i = 0; i < arr->size() - 1; i++) {
			file << arr->at(i) << sep;
		}
		file << arr->at(-1);
		file.close();
	}

	template <typename T>
	void save(std::string name, NDArray<T>* arr) {
		save(name, arr, ' ');
	}

	template <typename T>
	NDArray<T>* fromFunction(std::function<T(int)> func, int size) {
		NDArray<T>* out = new NDArray<T>(size);
		for (int i = 0; i < size; i++) {
			out->at(i) = func(i);
		}
		return out;
	}
}