#include "Test.cuh"

void testCreation() {
	Shape* shap = new Shape(new int[] {2, 2, 2}, 2);
	Array<int>* arr = ArrayOps<int>::ones(shap);
	// cout << arr->toString() << endl;

	auto a = ArrayOps<double>::arange(1, 5, 0.2);
	// a->print();
	auto* b = ArrayOps<double>::arange(5.4, 0.3);
	// b->print();
	auto c = ArrayOps<double>::arange(6.9f);
	// c->reshape(new Shape(new int[] {2, 3}, 2))->print();
	// c->print();

	// ArrayOps<int>::empty(new Shape(new int[] {2, 2, 2, 2}, 3))->print();
	// ArrayOps<double>::emptyLike(a)->print();
	// ArrayOps<int>::eye(5, 2)->print();
	// ArrayOps<float>::linspace(2.f, 3.f, 5.f)->print();

	auto d = ArrayOps<int>::arange(20);
	(d->reshape(new Shape(new int[] {4, 5}, 2)))->print();

	/*
	int fr[]{ 1, 2, 3, 4 };
	int er[]{ 1, 2, 3, 4, 5};
	Array<int> f = *ArrayOps<int>::asarray(fr, new Shape(new int[] {4}, 1));
	Array<int> e = *ArrayOps<int>::asarray(er, new Shape(new int[] {5}, 1));
	Array<int>* meshes = ArrayOps<int>::meshgrid(&f, &e);
	f.print();
	e.print();
	meshes[0].print();
	meshes[1].print();
	*/
}

void test() {
	testCreation();
}