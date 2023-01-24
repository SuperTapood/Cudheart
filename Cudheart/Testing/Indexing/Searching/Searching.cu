#include "Searching.cuh"

namespace Cudheart::Testing::Indexing::Searching {
	using namespace Cudheart::Searching;
	void test() {
		testArgmax();
		testArgmin();
		testNonzero();
		testArgwhere();
		testFlatnonzero();
		testWhere();
		testSearchsorted();
		testExtract();
		testCount_nonzero();
	}

	void testArgmax() {
		string cmd;

		int arr[] = {5, 2, 25, 25, 22, 21, 4, 7, 11};

		auto vec = new Vector(arr, 9);
		auto out = argmax(vec);

		cmd = Numpy::createArray("[5, 2, 25, 25, 22, 21, 4, 7, 11]", "vec");
		cmd += Numpy::argmax("vec", "res");

		Testing::submit("Searching::argmax(Vector<int>)", cmd, to_string(out));

	}

	void testArgmin() {
		string cmd;

		int arr[] = { 5, 2, 25, 25, 22, 21, 4, 7, 11 };

		auto vec = new Vector(arr, 9);
		auto out = argmin(vec);

		cmd = Numpy::createArray("[5, 2, 25, 25, 22, 21, 4, 7, 11]", "vec");
		cmd += Numpy::argmin("vec", "res");

		Testing::submit("Searching::argmin(Vector<int>)", cmd, to_string(out));
	}

	void testNonzero() {
		string cmd;

		int arr[] = { 5, 0, 25, 0, 22, 0, 4, 0, 11 };

		auto vec = new Vector(arr, 9);
		auto a = nonzero(vec);

		cmd = Numpy::createArray("[5, 0, 25, 0, 22, 0, 4, 0, 11]", "vec");
		cmd += Numpy::nonzero("vec", "res");

		Testing::submit("Searching::nonzero(Vector<int>)", cmd, a->toString());

		auto mat = (Matrix<int>*)vec->reshape(new Shape(3, 3));
		auto b = nonzero(mat);

		cmd = Numpy::createArray("[5, 0, 25, 0, 22, 0, 4, 0, 11]", "vec");
		cmd += Numpy::reshape("vec", "(3, 3)", "mat");
		cmd += Numpy::nonzero("mat", "res");

		Testing::submit("Searching::nonzero(Matrix<int>)", cmd, b->toString());
	}

	void testArgwhere() {
		string cmd;
		
		auto vec = arange(6, 2, 3)->flatten();
		auto a = argwhere(vec);

		cmd = Numpy::arange("0", "6", "1", "int", "vec");
		cmd += Numpy::argwhere("vec", "res");

		Testing::submit("Searching::argwhere(Vector<int>)", cmd, a->toString());

		auto mat = arange(6, 2, 3);
		auto b = argwhere(mat);

		cmd = Numpy::arange("0", "6", "1", "int", "vec");
		cmd += Numpy::reshape("vec", "(2, 3)", "mat");
		cmd += Numpy::argwhere("mat", "res");

		Testing::submit("Searching::argwhere(Matrix<int>)", cmd, b->toString());
	}

	void testFlatnonzero() {
		string cmd;

		int arr[] = { 5, 0, 25, 0, 22, 0, 4, 0, 11 };

		auto vec = new Vector(arr, 9);
		auto a = flatnonzero(vec);

		cmd = Numpy::createArray("[5, 0, 25, 0, 22, 0, 4, 0, 11]", "vec");
		cmd += Numpy::flatnonzero("vec", "res");

		Testing::submit("Searching::flatnonzero(Vector<int>)", cmd, a->toString());

		auto mat = (Matrix<int>*)vec->reshape(new Shape(3, 3));
		auto b = flatnonzero(mat);

		cmd = Numpy::createArray("[5, 0, 25, 0, 22, 0, 4, 0, 11]", "vec");
		cmd += Numpy::reshape("vec", "(3, 3)", "mat");
		cmd += Numpy::flatnonzero("mat", "res");

		Testing::submit("Searching::flatnonzero(Matrix<int>)", cmd, b->toString());
	}

	void testWhere() {
		string cmd;

		auto condVec = new Vector({ true, false, true, true, false, false, true, false});
		auto a = arange(8, 4, 2)->flatten();
		auto b = arange(14, 22, 1, 8, 1)->flatten();

		auto c = where(condVec, a, b);

		cmd = Numpy::createArray("[true, false, true, true, false, false, true, false]", "cond");
		cmd += Numpy::arange("0", "8", "1", "int", "a");
		cmd += Numpy::arange("14", "22", "1", "int", "b");
		cmd += Numpy::where("cond", "a", "b", "res");

		Testing::submit("Searching::where(Vector<bool>, Vector<int>, Vector<int>)", cmd, c->toString());
	}

	void testSearchsorted() {
		string cmd, add;

		auto vec = new Vector({ 1,2,3,4,5 });
		auto v = new Vector({ -10, 10, 2, 3 });
		auto sorter = arange(5, 1, 5)->flatten();

		cmd = Numpy::arange("1", "6", "1", "int", "vec");
		cmd += Numpy::createArray("[-10, 10, 2, 3]", "v");
		cmd += Numpy::arange("0", "5", "1", "int", "sorter");
		
		auto a = searchsorted(vec, v->get(0), "left", sorter);
		add = Numpy::searchsorted("vec", "v[0]", "'left'", "sorter", "res");
		Testing::submit("Searching::searchsorted(Vector<int>, int, 'left', Vector<int>)", cmd + add, to_string(a));

		auto ar = searchsorted(vec, v->get(0), "right", sorter);
		add = Numpy::searchsorted("vec", "v[0]", "'right'", "sorter", "res");
		Testing::submit("Searching::searchsorted(Vector<int>, int, 'right', Vector<int>)", cmd + add, to_string(ar));

		auto b = searchsorted(vec, v->get(0), "left");
		add = Numpy::searchsorted("vec", "v[0]", "'left'", "None", "res");
		Testing::submit("Searching::searchsorted(Vector<int>, int, 'left')", cmd + add, to_string(b));
		
		auto br = searchsorted(vec, v->get(0), "right");
		add = Numpy::searchsorted("vec", "v[0]", "'left'", "None", "res");
		Testing::submit("Searching::searchsorted(Vector<int>, int, 'left')", cmd + add, to_string(br));
		
		auto c = searchsorted(vec, v->get(0), sorter);
		add = Numpy::searchsorted("vec", "v[0]", "'left'", "sorter", "res");
		Testing::submit("Searching::searchsorted(Vector<int>, int, Vector<int>)", cmd + add, to_string(c));
		
		auto d = searchsorted(vec, v->get(0));
		add = Numpy::searchsorted("vec", "v[0]", "'left'", "None", "res");
		Testing::submit("Searching::searchsorted(Vector<int>, int)", cmd + add, to_string(d));
		
		auto e = searchsorted(vec, v, "left", sorter);
		add = Numpy::searchsorted("vec", "v", "'left'", "sorter", "res");
		Testing::submit("Searching::searchsorted(Vector<int>, Vector<int>, 'left', Vector<int>)", cmd + add, e->toString());
		
		auto er = searchsorted(vec, v, "right", sorter);
		add = Numpy::searchsorted("vec", "v", "'right'", "sorter", "res");
		Testing::submit("Searching::searchsorted(Vector<int>, Vector<int>, 'right', Vector<int>)", cmd + add, er->toString());
		
		auto f = searchsorted(vec, v, "left");
		add = Numpy::searchsorted("vec", "v", "'left'", "None", "res");
		Testing::submit("Searching::searchsorted(Vector<int>, Vector<int>, 'left')", cmd + add, f->toString());
		
		auto fr = searchsorted(vec, v, "right");
		add = Numpy::searchsorted("vec", "v", "'right'", "None", "res");
		Testing::submit("Searching::searchsorted(Vector<int>, Vector<int>, 'right')", cmd + add, fr->toString());
		
		auto g = searchsorted(vec, v, sorter);
		add = Numpy::searchsorted("vec", "v", "'left'", "sorter", "res");
		Testing::submit("Searching::searchsorted(Vector<int>, Vector<int>, Vector<int>)", cmd + add, g->toString());
		
		auto h = searchsorted(vec, v);
		add = Numpy::searchsorted("vec", "v", "'left'", "None", "res");
		Testing::submit("Searching::searchsorted(Vector<int>, Vector<int>)", cmd + add, h->toString());
	}

	void testExtract() {
		string cmd;

		auto cond = new Vector({ true, false, false, true, false, true, true, false, true });
		auto vec = arange(cond->getSize(), cond->getSize(), 1)->flatten();
		auto res = extract(cond, vec);

		cmd = Numpy::createArray("[true, false, false, true, false, true, true, false, true]", "cond");
		cmd += Numpy::arange("0", to_string(cond->getSize()), "1", "int", "vec");
		cmd += Numpy::extract("cond", "vec", "res");

		Testing::submit("Searching::extract(Vector<int>, Vector<int>)", cmd, res->toString());
	}

	void testCount_nonzero() {
		string cmd;

		auto v = new Vector({ 0, 1, 7, 0, 3, 0, 2, 19 });
		int i = count_nonzero(v);
		cmd = Numpy::createArray("[0, 1, 7, 0, 3, 0, 2, 19]", "v");
		cmd += Numpy::count_nonzero("v", "res");

		Testing::submit("Searching::count_nonzero(Vector<int>)", cmd, to_string(i));
	}
}