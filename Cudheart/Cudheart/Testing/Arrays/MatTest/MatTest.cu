#include "MatTest.cuh"

namespace Cudheart::Testing::Arrays::MatTest {
	void test() {
		testConstructors();
	}

	void testConstructors() {
		using Cudheart::NDArrays::Matrix;

		int* arr = new int[]{ 25, 25, 45, 588, 655555, 55, 568, 58, 999 };

		auto a = new Matrix<int>(5, 5);

		check("Matrix(int, int)", a->getShape()->toString() == "(5, 5)");

		for (int i = 0; i < 5; i++) {
			for (int j = 0; j < 5; j++) {
				a->get(i, j);
			}
		}

		auto b = new Matrix(arr, 4, 2);
		auto c = new Matrix(arr, 2, 4);

		for (int i = 0; i < 8; i++) {
			check("Matrix(int*, int, int)", b->get(i) == c->get(i));
		}
	}
}