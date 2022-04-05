#include "Test.cuh"

using namespace Cudheart;

void testVectorCreation() {
	// lol no need for using namespace when using auto

	auto arr = VectorOps::ones<int>(2);
	// cout << arr->toString() << endl;

	auto a = VectorOps::arange<double>(1, 5, 0.2);
	a->print();
	auto b = VectorOps::arange<double>(5.4, 0.3);
	b->print();
	auto c = VectorOps::arange<double>(6.9f);
	c->print();

	VectorOps::empty<int>((30))->print();
	VectorOps::emptyLike<double>(a)->print();
	VectorOps::linspace<float>(2.f, 3.f, 5.f)->print();

	auto d = VectorOps::arange<int>(20);
	d->print();
	auto e = c->castTo<long>();
	e->print();

	auto f = MatrixOps::fromVector<long>(e, 3, 2);
	f->print();
	auto g = MatrixOps::fromVector<long>(e, 2, 3);
	g->print();
}

void testMatrixCreation() {

}

void test() {
	testVectorCreation();
	testMatrixCreation();
}