#include "Cudheart/Cudheart.h"


#include <iostream>

using namespace std;
using namespace Cudheart::Test;

int main() {
	printf("using version %s\n", Cudheart::version.c_str());
	Cudheart::Test::test();
}