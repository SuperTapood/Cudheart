#include "Test.h"
#include "../Cudheart.h"


namespace Cudheart::Test {
	using Cudheart::Arrays::Array;
	using Cudheart::Arrays::ArrayOps;
	using Cudheart::Arrays::Shape;

	void TestCreation() {
		auto a = ArrayOps<double>::arange(1, 5, 0.2);
		print(a->toString());
		auto* b = ArrayOps<double>::arange(5.4, 0.3);
		print(b->toString());
		auto c = ArrayOps<double>::arange(7.9f);
		print(c->toString());
		print(ArrayOps<int>::empty(new Shape(new int[] {2, 2, 2, 2}, 3))->toString());
		print(ArrayOps<double>::emptyLike(a)->toString());
		print(ArrayOps<int>::eye(5, 2)->toString());
		print(ArrayOps<float>::linspace(2.f, 3.f, 5.f)->toString());

		int fr[]{ 1, 2, 3, 4 };
		int er[]{ 1, 2, 3, 4, 5 };
		Array<int> f = *ArrayOps<int>::asarray(fr, new Shape(new int[]{4}, 1));
		Array<int> e = *ArrayOps<int>::asarray(er, new Shape(new int[]{5}, 1));
		Array<int>* meshes = ArrayOps<int>::meshgrid(&f, &e);
	}

	void Test() {
		TestCreation();
	}

	void Assert(bool exp, string a, string b) {
		if (!exp) {
			throw AssertionError(a, b);
		}
	}

	void print(auto msg) {
		cout << msg << endl;
	}
}