#include "Util.h"
#include "Cudheart/Cudheart.h"

int main()
{
    Array<int> arr = *ArrayBuilder<int>::empty({ 5, 5, 5 });
    cout << arr << endl;
}