#include "VecTest.cuh"

namespace Cudheart::Testing::Arrays::VecTest {
	void test() {
		testConstructors();
		testCastTo();
		testReshape();
	}

	void testConstructors() {
		int arr[] = { 5, 7, 451, 14, 25, 250, 52205, 255, 897 };

		string sarr = "[5, 7, 451, 14, 25, 250, 52205, 255, 897]";

		Vector<int>* vec = new Vector<int>(arr, 9);

		string cmd = Numpy::createArray(sarr, "res");

		Testing::submit("Vector<int>(int[], int)", cmd, vec->toString());

		vec = new Vector<int>({ 5, 7, 451, 14, 25, 250, 52205, 255, 897 });

		Testing::submit("Vector<int>(int{})", cmd, vec->toString());
	}

	void testCastTo() {
		// idrk what to do with deez
		/*int arr[] = { 5, 7, 451, 14, 25, 250, 52205, 255, 897 };
		Vector<int>* a = new Vector<int>(arr, 9);

		Vector<StringType*>* b = a->castTo<StringType*>();

		check("int to string cast (Vector)", a->toString(), b->toString());

		Vector<ComplexType*>* c = a->castTo<ComplexType*>();
		Vector<ComplexType*>* d = b->castTo<ComplexType*>();

		check("int and string to complex (Vector)", c->toString(), d->toString());

		Vector<ComplexType*>* e = c->castTo<StringType*>()->castTo<ComplexType*>();

		check("complex to string to complex (Vector)", c->toString(), e->toString());

		Vector<float>* f = a->castTo<float>();

		for (int i = 0; i < f->getSize(); i++) {
			f->set(i, f->get(i) + 0.22);
		}

		check("complex to string to complex (Vector)", f->castTo<int>()->toString(), a->toString());*/
	}

	void testReshape() {
		int arr[] = { 5, 7, 451, 14, 25, 250 };
		Vector<int>* a = new Vector<int>(arr, 6);

		auto b = a->reshape(new Shape(2, 3));

		string cmd = Numpy::createArray("[5, 7, 451, 14, 25, 250]", "a");
		cmd += Numpy::reshape("a", "(2, 3)", "res");

		Testing::submit("Vector<int>->reshape((2, 3))", cmd, b->toString());
	}
}