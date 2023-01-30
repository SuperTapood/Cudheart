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
		testAugment();
	}

	void testConstructors() {
		using Cudheart::NDArrays::Matrix;
		string cmd;

		int arr[] = { 25, 25, 45, 588, 655555, 55, 568, 58, 999 };

		auto b = new Matrix(arr, 4, 2);

		cmd = Numpy::createArray("[25, 25, 45, 588, 655555, 55, 568, 58]", "vec");
		cmd += Numpy::reshape("vec", "(4, 2)", "res");

		Testing::submit("Matrix(int, int)", cmd, b->toString());

		auto c = new Matrix({ 25, 25, 45, 588, 655555, 55, 568, 58, 999 }, 3, 3);

		cmd = Numpy::createArray("[25, 25, 45, 588, 655555, 55, 568, 58, 999]", "vec");
		cmd += Numpy::reshape("vec", "(3, 3)", "res");

		Testing::submit("Matrix(int{}, int)", cmd, c->toString());
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
		string cmd;

		Matrix<int>* a = new Matrix<int>(arr, 2, 3);

		auto b = a->reshape<int>(new Shape(6));

		cmd = Numpy::createArray("[5, 7, 451, 14, 25, 250]", "res");

		Testing::submit("Matrix<int>->reshape((6))", cmd, b->toString());
	}

	void testReverseRows() {
		int arr[] = { 5, 7, 451, 14, 25, 250, 52205, 255, 897 };
		string cmd;

		Matrix<int>* a = new Matrix<int>(arr, 3, 3);

		auto b = a->reverseRows();

		cmd = Numpy::createArray("[5, 7, 451, 14, 25, 250, 52205, 255, 897]", "(3, 3)", "mat");
		cmd += "res = np.flip(mat, 1)\n";

		Testing::submit("Matrix<int>->reverseRows()", cmd, b->toString());
	}

	void testReverseCols() {
		int arr[] = { 5, 7, 451, 14, 25, 250, 52205, 255, 897 };
		string cmd;

		Matrix<int>* a = new Matrix<int>(arr, 3, 3);

		auto b = a->reverseCols();

		cmd = Numpy::createArray("[5, 7, 451, 14, 25, 250, 52205, 255, 897]", "(3, 3)", "mat");
		cmd += "res = np.flip(mat, 0)\n";

		Testing::submit("Matrix<int>->reverseCols()", cmd, b->toString());
	}

	void testTranspose() {
		int arr[] = { 5, 7, 451, 14, 25, 250, 52205, 255, 897 };
		string cmd;

		Matrix<int>* a = new Matrix<int>(arr, 3, 3);

		auto b = (Matrix<int>*)a->transpose();

		cmd = Numpy::createArray("[5, 7, 451, 14, 25, 250, 52205, 255, 897]", "(3, 3)", "mat");
		cmd += Numpy::T("mat", "res");

		Testing::submit("Matrix<int>->transpose()", cmd, b->toString());
	}

	void testRot90() {
		int arr[] = { 5, 7, 451, 14, 25, 250, 52205, 255, 897 };
		string cmd = Numpy::createArray("[5, 7, 451, 14, 25, 250, 52205, 255, 897]", "(3, 3)", "mat");
		string add;

		Matrix<int>* a = new Matrix<int>(arr, 3, 3);

		auto b = a->rot90(1);

		add = Numpy::rot90("mat", 1, "res");

		Testing::submit("Matrix<int>->rot90(k=1)", cmd + add, b->toString());

		b = a->rot90(2);

		add = Numpy::rot90("mat", 2, "res");

		Testing::submit("Matrix<int>->rot90(k=2)", cmd + add, b->toString());

		b = a->rot90(3);

		add = Numpy::rot90("mat", 3, "res");

		Testing::submit("Matrix<int>->rot90(k=3)", cmd + add, b->toString());

		add = Numpy::rot90("mat", 4, "res");

		Testing::submit("Matrix<int>->rot90(k=4)", cmd + add, a->rot90(4)->toString());
	}

	void testAugment() {
		int arr[] = { 5, 7, 451, 14, 25, 250, 52205, 255, 897 };
		string cmd;

		Matrix<int>* mat = new Matrix<int>(9, 2);

		for (int i = 0; i < 9; i++) {
			for (int j = 0; j < 2; j++) {
				mat->set(i, j, arr[i]);
			}
		}

		Vector<int>* vec = new Vector<int>(arr, 9);

		Matrix<int>* mat2 = mat->augment(vec);

		cmd = Numpy::createArray("[5, 7, 451, 14, 25, 250, 52205, 255, 897]", "vec");
		cmd += Numpy::createArray(mat->toString(), "mat");
		cmd += Numpy::createArray("[5, 7, 451, 14, 25, 250, 52205, 255, 897]", "(9, 1)", "mat2");
		cmd += Numpy::augment("mat", "mat2", 1, "res");

		Testing::submit("Matrix<int>->augment(Vector<int>)", cmd, mat2->toString());
	}
}