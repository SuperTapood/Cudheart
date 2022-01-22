#include "Test.h"

void Test::test() {
	int arr[] = { 5, 4, 7, 252, 5270 };
	int* a = (int*)arr;
	VectorInt vec = ArrayOps::asarray(a, 5);
	cout << vec.toString() << endl;
}