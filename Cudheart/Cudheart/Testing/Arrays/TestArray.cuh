#pragma once
#include <stdio.h>
#include "../TestUtil.cuh"

namespace Cudheart::Testing {
	void testArrays();
}


namespace Cudheart::Testing::Arrays::ArrayOps{
	void test();

	void testAppend();

	void testConcatenate();

	void testSplit();

	void testTile();

	void testRemove();

	void testTrimZeros();

	void testUnique();
}

namespace Cudheart::Testing::Arrays::IO {
	void test();

	void testFromString();

	void testSave();

	void testFromFile();

	void testFromFunction();
}


namespace Cudheart::Testing::Arrays::VecTest {
	void test();

	void testConstructors();

	void testCastTo();

	void testReshape();
}