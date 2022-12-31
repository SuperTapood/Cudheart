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

		auto d = new Matrix(new int[] { 25, 25, 45, 588, 655555, 55, 568, 58, 999 }, 5, 5);

		for (int i = 0; i < c->getSize(); i++) {
			check("Matrix(int{}, int, int)", d->get(i) == c->get(i));
		}
	}

	void testCastTo() {
		int* arr = new int[] {5, 7, 451, 14, 25, 250, 52205, 255, 897};
		Matrix<int>* a = new Matrix<int>(arr, 3, 3);

		Matrix<StringType*>* b = a->castTo<StringType*>();

		check("int to string cast (Matrix)", a->toString() == b->toString());

		Matrix<ComplexType*>* c = a->castTo<ComplexType*>();
		Matrix<ComplexType*>* d = b->castTo<ComplexType*>();

		check("int and string to complex (Matrix)", c->toString() == d->toString());

		Matrix<ComplexType*>* e = c->castTo<StringType*>()->castTo<ComplexType*>();

		check("complex to string to complex (Matrix)", c->toString() == e->toString());

		Matrix<float>* f = a->castTo<float>();

		for (int i = 0; i < f->getSize(); i++) {
			f->set(i, f->get(i) + 0.22);
		}

		check("complex to string to complex (Matrix)", f->castTo<int>()->toString() == a->toString());
	}

	void testReshape() {
		int* arr = new int[] {5, 7, 451, 14, 25, 250};
		Matrix<int>* a = new Matrix<int>(arr, 2, 3);

		auto b = a->reshape<int>(new Shape(6));

		for (int i = 0; i < b->getSize(); i++) {
			check("reshape Matrix -> Vector", a->get(i) == b->get(i));
		}
	}

	void testReverseRows() {
		int* arr = new int[] {5, 7, 451, 14, 25, 250, 52205, 255, 897};
	}
}