#include "Test.cuh"
#include "../../Sample/kernel.cuh"

using namespace Cudheart;

void testVectorCreation() {
#pragma region test_ones
	auto arr = VectorOps::ones<int>(2);
	// cout << arr->toString() << endl;

	for (int i = 0; i < 2; i++) {
		assertTest("test ones", arr->get(i), 1);
	}
#pragma endregion

#pragma region test_arange
	auto a = VectorOps::arange<double>(1, 5, 0.2);

	double dv = 1;
	for (int i = 0; i < a->getSize(); i++) {
		assertTest("test arange(1, 5, 0.2)", a->get(i), dv);
		dv += 0.2;
	}

	auto b = VectorOps::arange<double>(5.4, 0.3);

	dv = 0;
	for (int i = 0; i < b->getSize(); i++) {
		assertTest("test arange(5.4, 0.3)", b->get(i), dv);
		dv += 0.3;
	}
	auto c = VectorOps::arange<double>(6.9f);

	for (int i = 0; i < c->getSize(); i++) {
		assertTest("test arange(6.9f)", c->get(i), i);
	}
#pragma endregion

#pragma region test_empty
	VectorOps::empty<int>((30));
	VectorOps::emptyLike<double>(a);
#pragma endregion

#pragma region test_linspace
	auto v = VectorOps::linspace<float>(2.f, 3.f, 5.f);
	for (int i = 0; i < v->getSize(); i++) {
		assertTest("test linspace(2.f, 3.f, 5.f)", v->get(i), 2.f + i * ((3.f - 2.f) / 4.f));
	}
#pragma endregion

#pragma region test_casting
	auto d = VectorOps::arange<double>(20.0);
	auto e = d->castTo<long>();
#pragma endregion

#pragma region test_fromVector
	auto f = MatrixOps::fromVector<long>(e, 10, 2);
	// f->print();
	auto g = MatrixOps::fromVector<long>(e, 5, 4);
	// g->print();
#pragma endregion
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

	Cudheart::NDArrays::Matrix<int>** f = MatrixOps::meshgrid<int, int, int>(d, e);
	//f[0].print();
	//f[1].print();

	Cudheart::NDArrays::Matrix<int>* g = f[0];

	auto h = MatrixOps::diag<int>(g, 2);
	// h->print();
	auto i = MatrixOps::diagflat<int>(h, 2);
	// i->print();

	auto j = MatrixOps::tri<int>(3, 5, 2);
	// j->print();

	auto k = MatrixOps::arange(12, 4, 3);
	// k->print();
	//f->print();
	auto l = f[0]->transpose();
	//l->print();
	auto m = f[0]->transpose()->transpose();
	//m->print();
	auto n = f[0]->reverseRows();
	// n->print();
	auto o = f[0]->rotate(90);
	// o->print();
	auto p = f[0]->rotate(180);
	// p->print();
	auto q = f[0]->rotate(-180);
	// q->print();
	auto r = f[0]->rotate(-90);
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
	using namespace Cudheart::CPP::Math;
	using namespace Cudheart::CPP::Math::Exp;
	using namespace Cudheart::CPP::Math::Trigo;

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
	// cout << m << endl;
}

void testLogicOpsCPP() {
	using namespace Cudheart::Logic;
	auto a = VectorOps::arange(9);
	auto b = MatrixOps::fromVector(a, 3, 3);
	auto c = VectorOps::zeros<int>(9);
	auto d = MatrixOps::fromVector(c, 3, 3);

	//cout << all(a) << endl;
	//cout << all(b) << endl;
	//cout << all(c) << endl;
	//cout << all(d) << endl;
	//cout << any(a) << endl;
	//cout << any(b) << endl;
	//cout << any(c) << endl;
	//cout << any(d) << endl;
	//cout << 0 && 0;
	//cout << endl;
	//logicalAnd(a, c)->print();
	//logicalOr(a, c)->print();
}

void testExceptions() {
	// AssertionException("testy festy", 69, 420).raise();
	// BadValueException("testy festy", to_string(69), "biddy dib bid").raise();
	// BaseException("message").raise();
	// CudaException(cudaError::cudaErrorAssert, "message").raise();
	// CudaException(cudaError::cudaErrorMemoryAllocation, "messgae").raise();
	// IndexOutOfBoundsException(6, 6, 1).raise();
	// IndexOutOfBoundsException(698, 104, 205, 205).raise();
	// IndexOutOfBoundsException(69, 420).raise();
	// MatrixConversionException(5, 5, 5).raise();
	// NotImplementedException("name", "dep").raise();
	// ShapeMismatchException(20, 1040, 205, 12).raise();
	// ShapeMismatchException("custom").raise();
	// ShapeMismatchException(54, 41).raise();
	// ZeroDivisionException("funcy muncy").raise();
}

void test() {
	testExceptions();
	testVectorCreation();
	testMatrixCreation();
	testBinaryOpsCPP();
	testBinaryOpsCuda();
	testLinalgOpsCPP();
	testLogicOpsCPP();
}