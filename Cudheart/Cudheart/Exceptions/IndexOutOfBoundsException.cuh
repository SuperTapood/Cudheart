#pragma once

#include "BaseException.cuh"

namespace Cudheart::Exceptions {
	class IndexOutOfBoundsException : public BaseException {
	public:
		IndexOutOfBoundsException(int len, int got) {
			m_msg = "Index Out Of Bounds:"
				"index " + std::to_string(got) +
				" exceeds vector len " + std::to_string(got) + ".";
			raise();
		}

		IndexOutOfBoundsException(int width, int height, int i, int j) {
			m_msg = "Index Out Of Bounds: "
				"index " + std::to_string(i) + " and jdex " + std::to_string(j) +
				" exceeds matrix dims (" + std::to_string(height) + "," + std::to_string(width) + ").";
			raise();
		}

		IndexOutOfBoundsException(int width, int height, int i) {
			m_msg = "Index Out Of Bounds:"
				"index " + std::to_string(i) +
				" exceeds matrix dims (" + std::to_string(height) + "," + std::to_string(width) + ").";
			raise();
		}
	};
}