#include "VecTest.cuh"

namespace Cudheart::Testing::Arrays::VecTest {
	void test() {
		testConstructors();
		testCopy();
		testCastTo();
		testReshape();
	}

	void testConstructors() {
		int arr[] = { 5, 7, 451, 14, 25, 250, 52205, 255, 897 };

		Vector<int>* vec = new Vector<int>(arr, 9);

		for (int i = 0; i < 9; i++) {
			check("Vector<int>(int*, int)", vec->get(i), arr[i]);
		}

		vec = new Vector<int>(7);
		for (int i = 0; i < 7; i++) {
			check("Vector<int>(int)", vec->get(i), vec->get(i));
		}

		vec = new Vector<int>({ 5, 7, 451, 14, 25, 250, 52205, 255, 897 });
		for (int i = 0; i < 9; i++) {
			check("Vector<int>(initializer_list<int>)", vec->get(i), arr[i]);
		}
	}

	void testCastTo() {
		int arr[] = { 5, 7, 451, 14, 25, 250, 52205, 255, 897 };
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

		check("complex to string to complex (Vector)", f->castTo<int>()->toString(), a->toString());
	}

	void testReshape() {
		int arr[] = { 5, 7, 451, 14, 25, 250 };
		Vector<int>* a = new Vector<int>(arr, 6);

		auto b = a->reshape<int>(new Shape(2, 3));

		for (int i = 0; i < a->getSize(); i++) {
			check("reshape vector -> matrix", a->get(i), b->get(i));
		}
	}

	void testCopy() {
		float arr[] = { 5.5, 7.5, 451.5, 14.5, 25.5, 250.5, 41.5, 58.5 };
		Vector<float>* a = new Vector<float>(arr, 8);
		Vector<float>* b = (Vector<float>*)a->copy();

		for (int i = 0; i < 8; i++) {
			check("copy<T = int>()", a->get(i) == b->get(i) && b->get(i) == arr[i]);
		}
	}
}