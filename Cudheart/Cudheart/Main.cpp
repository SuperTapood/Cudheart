#include "Cudheart/test.h"
#include "Cudheart/Cudheart.h"
#include "Cudheart/Test/Test.h"

#if defined _DEBUG
    #define debug true
#else
    #define debug false
#endif

int main()
{
    // Test::test();
    if (!debug) {
        string s;
        cin >> s;
    }
}