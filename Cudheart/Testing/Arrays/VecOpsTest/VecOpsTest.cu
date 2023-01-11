#include "VecOpsTest.cuh"

namespace Cudheart::Testing::Arrays::VecOpsTest {
	using namespace Cudheart::VectorOps;

	void test() {
		testEmptyLike();
		testArange();
	}

	void testEmptyLike() {
		auto a = new Vector<int>(5);
		auto b = emptyLike<int>(a);

		check("VectorOps::emptyLike<int>", a->getShape()->toString(), b->getShape()->toString());
	}

	void testArange() {
		auto a = arange(5);

		for (int i = 0; i < a->getSize(); i++) {
			check("VectorOps::arange<int>(int)", a->get(i), i);
		}

		auto b = arange(-1);
		check("VectorOps::arange<int>(int)", b->getShape()->toString(), "(0,)");

		auto c = arange(1, 5);
		check("VectorOps::arange<int>(int)", c->getShape()->toString(), "(0,)");

		auto d = arange(7, 5, 5);
		check("VectorOps::arange<int>(int)", d->getShape()->toString(), "(0,)");

		auto e = arange(5, 2);
		for (int i = 0, idx = 0; i < 5; i += 2, idx++) {
			check("VectorOps::arange<int>(int, int)", e->get(idx) == i);
		}

		auto f = arange(3, 7, 1);
		for (int i = 3, idx = 0; i < 7; i++, idx++) {
			check("VectorOps::arange<int>(int, int)", f->get(idx) == i);
		}
	}
}