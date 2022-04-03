#include "Test.cuh"

void testCreation() {
	Vector<int>* arr = VectorOps<int>::ones(2);
	// cout << arr->toString() << endl;

	auto a = VectorOps<double>::arange(1, 5, 0.2);
	a->print();
	auto* b = VectorOps<double>::arange(5.4, 0.3);
	b->print();
	auto c = VectorOps<double>::arange(6.9f);
	c->print();

	VectorOps<int>::empty((30))->print();
	VectorOps<double>::emptyLike(a)->print();
	VectorOps<float>::linspace(2.f, 3.f, 5.f)->print();

	auto d = VectorOps<int>::arange(20);
	d->print();
}

void test() {
	testCreation();
}