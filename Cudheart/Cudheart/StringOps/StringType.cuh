#pragma once

#include <string>

namespace Cudheart {
	class StringType {
	private:
		std::string str;
		
	public:
		StringType(std::string string) {
			str = string;
		}
	};
}