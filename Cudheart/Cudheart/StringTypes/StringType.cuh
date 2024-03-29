#pragma once

#include <string>

using namespace std;

namespace Cudheart {
	class StringType {
	public:
		std::string m_str;

	public:
		StringType(std::string string) {
			m_str = string;
		}

		std::string str() {
			return m_str;
		}

		string toString() {
			return m_str;
		}

		long double toFloating() {
			return std::stold(m_str);
		}

		const char* c_str() {
			return m_str.c_str();
		}
	};
}