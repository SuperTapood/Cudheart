#include "Test.h"

void Test::test() {
	Vector<int> v = Arrays::arange<int>(0., 5., 1.);
	cout << v.toString();
}