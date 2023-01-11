#include "MatTest.cuh"

namespace Cudheart::Testing::Arrays::MatTest {
	void test() {
		testConstructors();
		testCastTo();
		testReshape();
		testReverseRows();
		testReverseCols();
		testTranspose();
		testRot90();
	}

	void testConstructors() {
		using Cudheart::NDArrays::Matrix;

		int arr[] = { 25, 25, 45, 588, 655555, 55, 568, 58, 999 };

		auto a = new Matrix<int>(3, 3);

		check("Matrix(int, int)", a->getShape()->toString(), "(3, 3)");

		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				a->get(i, j);
			}
		}

		auto b = new Matrix(arr, 4, 2);
		auto c = new Matrix(arr, 2, 4);

		for (int i = 0; i < 8; i++) {
			check("Matrix(int*, int, int)", b->get(i), c->get(i));
		}

		auto d = new Matrix({ 25, 25, 45, 588, 655555, 55, 568, 58, 999 }, 3, 3);

		for (int i = 0; i < c->getSize(); i++) {
			check("Matrix(int{}, int, int)", d->get(i), c->get(i));
		}
	}

	void testCastTo() {
		int arr[] = { 5, 7, 451, 14, 25, 250, 52205, 255, 897 };
		Matrix<int>* a = new Matrix<int>(arr, 3, 3);

		Matrix<StringType*>* b = a->castTo<StringType*>();

		check("int to string cast (Matrix)", a->toString(), b->toString());

		Matrix<ComplexType*>* c = a->castTo<ComplexType*>();
		Matrix<ComplexType*>* d = b->castTo<ComplexType*>();

		check("int and string to complex (Matrix)", c->toString(), d->toString());

		Matrix<ComplexType*>* e = c->castTo<StringType*>()->castTo<ComplexType*>();

		check("complex to string to complex (Matrix)", c->toString(), e->toString());

		Matrix<float>* f = a->castTo<float>();

		for (int i = 0; i < f->getSize(); i++) {
			f->set(i, f->get(i) + 0.22);
		}

		check("complex to string to complex (Matrix)", f->castTo<int>()->toString(), a->toString());
	}

	void testReshape() {
		int arr[] = { 5, 7, 451, 14, 25, 250 };
		Matrix<int>* a = new Matrix<int>(arr, 2, 3);

		auto b = a->reshape<int>(new Shape(6));

		for (int i = 0; i < b->getSize(); i++) {
			check("reshape Matrix -> Vector", a->get(i), b->get(i));
		}
	}

	void testReverseRows() {
		int arr[] = { 5, 7, 451, 14, 25, 250, 52205, 255, 897 };
		Matrix<int>* a = new Matrix<int>(arr, 3, 3);

		auto b = a->reverseRows();

		for (int i = 0; i < a->getHeight(); i++) {
			for (int j = 0; j < a->getWidth(); j++) {
				check("Matrix<int>->reverseRows()", a->get(i, j), b->get(i, a->getWidth() - j - 1));
			}
		}
	}

	void testReverseCols() {
		int arr[] = { 5, 7, 451, 14, 25, 250, 52205, 255, 897 };
		Matrix<int>* a = new Matrix<int>(arr, 3, 3);

		auto b = a->reverseCols();

		for (int i = 0; i < a->getHeight(); i++) {
			for (int j = 0; j < a->getWidth(); j++) {
				check("Matrix<int>->reverseCols()", a->get(i, j), b->get(a->getHeight() - i - 1, j));
			}
		}
	}

	void testTranspose() {
		int arr[] = { 5, 7, 451, 14, 25, 250, 52205, 255, 897 };
		Matrix<int>* a = new Matrix<int>(arr, 3, 3);

		auto b = (Matrix<int>*)a->transpose();

		for (int i = 0; i < a->getHeight(); i++) {
			for (int j = 0; j < a->getWidth(); j++) {
				check("Matrix<int>->transpose()", a->get(i, j), b->get(j, i));
			}
		}
	}

	void testRot90() {
		int arr[] = { 5, 7, 451, 14, 25, 250, 52205, 255, 897 };
		Matrix<int>* a = new Matrix<int>(arr, 3, 3);

		auto b = ((Matrix<int>*)(a->transpose()))->reverseCols();
		check("Matrix<int>->rot90(k=1)", a->rot90(1)->toString(), b->toString());
		b = ((Matrix<int>*)(b->transpose()))->reverseCols();
		check("Matrix<int>->rot90(k=2)", a->rot90(2)->toString(), b->toString());
		b = ((Matrix<int>*)(b->transpose()))->reverseCols();
		check("Matrix<int>->rot90(k=3)", a->rot90(3)->toString(), b->toString());
		check("Matrix<int>->rot90(k=4)", a->rot90(4)->toString(), a->toString());
	}

	void testAugment() {
		int arr[] = { 5, 7, 451, 14, 25, 250, 52205, 255, 897 };
		Matrix<int>* mat = new Matrix<int>(9, 2);

		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 9; j++) {
				mat->set(i, j, arr[j]);
			}
		}

		Vector<int>* vec = new Vector<int>(arr, 9);

		Matrix<int>* mat2 = mat->augment(vec);

		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 9; j++) {
				check("Matrix<int>->augment(Vector<int>)", mat2->get(i, j), arr[j]);
			}
		}
	}
}