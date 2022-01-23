#include "Test.h"
#include "../Exceptions/Exceptions.h"

void Test::test() {
    try {
        Vector v = ArrayOps::arange(5);
        cout << v;
        cout << v[6];
    }
    catch (BaseException& e) {
        std::cout << "MyException caught" << std::endl;
        std::cout << e.what() << std::endl;
    }
}