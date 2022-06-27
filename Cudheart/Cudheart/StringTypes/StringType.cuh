#pragma once

#include <string>

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

		const char* c_str() {
			return m_str.c_str();
		}
	};
}