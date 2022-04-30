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
	//f[0].transpose()->print();
	//f[0].transpose()->transpose()->print();
	//f[0].reverseRows()->print();
	//f->rotate(90)->print();
	//f->rotate(180)->print();
	//f->rotate(-180)->print();
	//f->rotate(-90)->print();
} 

void testBinaryOpsCuda() {
	using namespace Cudheart::CUDA::Math::Bitwise;

	auto a = VectorOps::full(4, 5);
	auto b = VectorOps::full(4, 9);
	auto c = VectorOps::full(4, 1);

	bitwiseAnd(a, b)->print();;
	bitwiseOr(a, b)->print();
	bitwiseXor(a, b)->print();
	bitwiseNot(a)->print();
	bitwiseLeftShift(b, c)->print();
	bitwiseRightShift(b, c)->print();
}

void testBinaryOpsCPP() {
	using namespace Cudheart::CPP::Math::Bitwise;

	auto a = VectorOps::full(4, 5);
	auto b = VectorOps::full(4, 9);
	auto c = VectorOps::full(4, 1);

	bitwiseAnd(a, b)->print();;
	bitwiseOr(a, b)->print();
	bitwiseXor(a, b)->print();
	bitwiseNot(a)->print();
	bitwiseLeftShift(b, c)->print();
	bitwiseRightShift(b, c)->print();
}

void testEMathOpsCPP() {
	using namespace Cudheart::CPP::Math::EMath;

	auto a = VectorOps::full<float>(25, 25);
	auto b = VectorOps::full(25, 0.5);
	auto c = MatrixOps::fromVector(b, 5, 5);

	//squareRoot(a)->print();
	//loga(a)->print();
	//loga2(a)->print();
	//logan(a, 5.f)->print();
	//loga10(a)->print();
	arccos(b)->print();
	arccos(c)->print();
	arcsin(b)->print();
	arcsin(c)->print();
	arctan(b)->print();
	arctan(c)->print();
	arccot(b)->print();
	arccot(c)->print();
}

void test() {
	//testVectorCreation();
	//testMatrixCreation();
	//testBinaryOpsCPP();
	//testBinaryOpsCuda();
	testEMathOpsCPP();
}