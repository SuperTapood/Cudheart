#include "Test.cuh"
#include "../../Sample/kernel.cuh"

using namespace Cudheart;

void testVectorCreation() {
	// lol no need for using namespace when using auto

	auto arr = VectorOps::ones<int>(2);
	// cout << arr->toString() << endl;

	auto a = VectorOps::arange<double>(1, 5, 0.2);
	// a->print();
	auto b = VectorOps::arange<double>(5.4, 0.3);
	// b->print();
	auto c = VectorOps::arange<double>(6.9f);
	// c->print();

	VectorOps::empty<int>((30));
	VectorOps::emptyLike<double>(a);
	VectorOps::linspace<float>(2.f, 3.f, 5.f);

	auto d = VectorOps::arange<int>(20);
	// d->print();
	auto e = c->castTo<long>();
	// e->print();

	auto f = MatrixOps::fromVector<long>(e, 3, 2);
	// f->print();
	auto g = MatrixOps::fromVector<long>(e, 2, 3);
	// g->print();
}

void testMatrixCreation() {
	auto a = MatrixOps::eye<int>(4);
	// a->print();
	auto b = MatrixOps::eye<int>(7, 5, 4);
	// b->print();
	auto c = MatrixOps::identity<long>(5);
	// c->print();

	auto d = VectorOps::arange(4);
	auto e = VectorOps::arange(5);

	Cudheart::NDArrays::Matrix<int>* f = MatrixOps::meshgrid<int, int, int>(d, e);
	//f[0].print();
	//f[1].print();

	Cudheart::NDArrays::Matrix<int>* g = &f[0];

	auto h = MatrixOps::diag<int>(g, 2);
	// h->print();
	auto i = MatrixOps::diagflat<int>(h, 2);
	// i->print();

	auto j = MatrixOps::tri<int>(3, 5, 2);
	// j->print();

	auto k = MatrixOps::arange(12, 4, 3);
	// k->print();
	//f->print();
	auto l = f[0].transpose();
	//l->print();
	auto m = f[0].transpose()->transpose();
	//m->print();
	auto n = f[0].reverseRows();
	// n->print();
	auto o = f->rotate(90);
	// o->print();
	auto p = f->rotate(180);
	// p->print();
	auto q = f->rotate(-180);
	// q->print();
	auto r = f->rotate(-90);
	// r->print();
}

void testBinaryOpsCuda() {
	using namespace Cudheart::CUDA::Math::Bitwise;

	auto a = VectorOps::full(4, 5);
	auto b = VectorOps::full(4, 9);
	auto c = VectorOps::full(4, 1);

	auto d = bitwiseAnd(a, b);
	// d->print();
	auto e = bitwiseOr(a, b);
	// e->print();
	auto f = bitwiseXor(a, b);
	// f->print();
	auto g = bitwiseNot(a);
	// g->print();
	auto h = bitwiseLeftShift(b, c);
	// h->print();
	auto i = bitwiseRightShift(b, c);
	// i->print();
}

void testBinaryOpsCPP() {
	using namespace Cudheart::CPP::Math::Bitwise;

	auto a = VectorOps::full(4, 5);
	auto b = VectorOps::full(4, 9);
	auto c = VectorOps::full(4, 1);

	auto d = bitwiseAnd(a, b);
	// d->print();
	auto e = bitwiseOr(a, b);
	// e->print();
	auto f = bitwiseXor(a, b);
	// f->print();
	auto g = bitwiseNot(a);
	// g->print();
	auto h = bitwiseLeftShift(b, c);
	// h->print();
	auto i = bitwiseRightShift(b, c);
	// i->print();
}

void testEMathOpsCPP() {
	using namespace Cudheart::CPP::Math::EMath;

	auto a = VectorOps::full<float>(25, 25);
	auto b = VectorOps::full(25, 0.5);
	auto c = MatrixOps::fromVector(b, 5, 5);

	auto d = squareRoot(a);
	// d->print();
	auto e = loga(a);
	// e->print();
	auto f = loga2(a);
	// f->print();
	auto g = logan(a, 5.f);
	// g->print();
	auto h = loga10(a);
	// h->print();
	auto i = arccos(b);
	// i->print();
	auto j = arccos(c);
	// j->print();
	auto k = arcsin(b);
	// k->print();
	auto l = arcsin(c);
	// l->print();
	auto m = arctan(b);
	// m->print();
	auto n = arctan(c);
	// n->print();
	auto o = arccot(b);
	// o->print();
	auto p = arccot(c);
	// p->print();
}

void testEMathOpsCuda() {
	using namespace Cudheart::CUDA::Math::EMath;

	auto a = VectorOps::full<float>(25, 25);
	auto b = VectorOps::full(25, 0.5);
	auto c = MatrixOps::fromVector(b, 5, 5);

	auto d = squareRoot(a);
	// d->print();
	auto e = loga(a);
	// e->print();
	auto f = loga2(a);
	// f->print();
	auto g = logan(a, 5.f);
	// g->print();
	auto h = loga10(a);
	// h->print();
	auto i = arccos(b);
	// i->print();
	auto j = arccos(c);
	// j->print();
	auto k = arcsin(b);
	// k->print();
	auto l = arcsin(c);
	// l->print();
	auto m = arctan(b);
	// m->print();
	auto n = arctan(c);
	// n->print();
	auto o = arccot(b);
	// o->print();
	auto p = arccot(c);
	// p->print();
}

void testLinalgOpsCPP() {
	using namespace Cudheart::CPP::Math::Linalg;
	auto a = VectorOps::arange(1, 5, 1);
	auto b = VectorOps::arange(1, 5, 1);
	auto c = VectorOps::arange(1, 17, 1);
	auto d = MatrixOps::fromVector(c, 4, 4);
	auto y = MatrixOps::fromVector(VectorOps::arange(1, 10, 1), 3, 3);
	auto z = MatrixOps::fromVector(b, 2, 2);

	auto e = dot(a, b);
	// cout << c << endl;
	auto f = dot(b, d);
	//f->print();
	auto g = dot(d, b);
	//g->print();

	auto h = inner(d, d);
	// h->print();

	auto i = outer(a, a);
	// i->print();

	auto j = outer(d, d);
	// j->print();

	auto k = outer(d, a);
	// k->print();

	auto l = outer(a, d);
	// l->print();
	// cout << det(d->castTo<long double>()) << endl;

	auto m = trace(MatrixOps::eye<int>(3));
	cout << m << endl;
}

void testLogicOpsCPP() {
	using namespace Cudheart::CPP::Logic;
	auto a = VectorOps::arange(9);
	auto b = MatrixOps::fromVector(a, 3, 3);
	auto c = VectorOps::zeros<int>(9);
	auto d = MatrixOps::fromVector(c, 3, 3);

	cout << all(a) << endl;
	cout << all(b) << endl;
	cout << all(c) << endl;
	cout << all(d) << endl;
	cout << any(a) << endl;
	cout << any(b) << endl;
	cout << any(c) << endl;
	cout << any(d) << endl;
	cout << 0 && 0;
	cout << endl;
	logicalAnd(a, c)->print();
}

void test() {
	// testVectorCreation();
	// testMatrixCreation();
	// testBinaryOpsCPP();
	// testBinaryOpsCuda();
	// testEMathOpsCPP();
	// testEMathOpsCuda();
	// testLinalgOpsCPP();
	testLogicOpsCPP();
}