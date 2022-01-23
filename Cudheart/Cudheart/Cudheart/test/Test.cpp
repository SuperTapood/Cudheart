#include "Test.h"
#include "../Exceptions/Exceptions.h"

void Test::test() {
    try {
        Vector v = ArrayOps::arange(5);
        cout << v;
        cout << v[6];
    }
    catch (BaseException& e) {
        e.print();
    }
}