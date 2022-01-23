#include "Test.h"
#include "../Exceptions/Exceptions.h"

void Test::test() {
    //int* a = new int[5]{ 1, 652, 652065, 520, 5 };
    //Vector vec = ArrayOps::asarray(a, 5, true);
    //cout << vec << endl;
    //a[3] = 25;
    //cout << vec << endl;
    try {
        Vector v = ArrayOps::arange(5);
        //cout << v;
        cout << v[6];
    }
    catch (BaseException& e) {
        std::cout << "MyException caught" << std::endl;
        std::cout << e.what() << std::endl;
    }
}