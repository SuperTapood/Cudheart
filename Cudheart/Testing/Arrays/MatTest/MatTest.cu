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

		check("Matrix(int, int)", a->toString());

		auto b = new Matrix(arr, 4, 2);

		check("Matrix(int, int)", b->toString());

		auto c = new Matrix({ 25, 25, 45, 588, 655555, 55, 568, 58, 999 }, 3, 3);

		check("Matrix(int{}, int)", c->toString());
	}

	void testCastTo() {
		// idrk what to do about this
		/*int arr[] = { 5, 7, 451, 14, 25, 250, 52205, 255, 897 };
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

		check("complex to string to complex (Matrix)", f->castTo<int>()->toString(), a->toString());*/
	}

	void testReshape() {
		int arr[] = { 5, 7, 451, 14, 25, 250 };
		Matrix<int>* a = new Matrix<int>(arr, 2, 3);

		auto b = a->reshape<int>(new Shape(6));

		check("reshape Matrix -> Vector", b->toString());
	}

	void testReverseRows() {
		int arr[] = { 5, 7, 451, 14, 25, 250, 52205, 255, 897 };
		Matrix<int>* a = new Matrix<int>(arr, 3, 3);

		auto b = a->reverseRows();

		check("Matrix<int>->reverseRows()", b->toString());
	}

	void testReverseCols() {
		int arr[] = { 5, 7, 451, 14, 25, 250, 52205, 255, 897 };
		Matrix<int>* a = new Matrix<int>(arr, 3, 3);

		auto b = a->reverseCols();

		check("Matrix<int>->reverseCols()", b->toString());
	}

	void testTranspose() {
		int arr[] = { 5, 7, 451, 14, 25, 250, 52205, 255, 897 };
		Matrix<int>* a = new Matrix<int>(arr, 3, 3);

		auto b = (Matrix<int>*)a->transpose();

		check("Matrix<int>->transpose()", b->toString());
	}

	void testRot90() {
		int arr[] = { 5, 7, 451, 14, 25, 250, 52205, 255, 897 };
		Matrix<int>* a = new Matrix<int>(arr, 3, 3);

		auto b = ((Matrix<int>*)(a->transpose()))->reverseCols();
		check("Matrix<int>->rot90(k=1)", b->toString());
		b = ((Matrix<int>*)(b->transpose()))->reverseCols();
		check("Matrix<int>->rot90(k=2)", b->toString());
		b = ((Matrix<int>*)(b->transpose()))->reverseCols();
		check("Matrix<int>->rot90(k=3)", b->toString());
		check("Matrix<int>->rot90(k=4)", a->rot90(4)->toString());
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

		check("Matrix<int>->augment(Vector<int>)", mat2->toString());
	}
}