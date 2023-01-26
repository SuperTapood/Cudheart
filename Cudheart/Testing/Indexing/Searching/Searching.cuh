#pragma once
#include "../../TestUtil.cuh"

namespace Cudheart::Testing::Indexing::Searching {
	void test();

	void testArgmax();

	void testArgmin();

	void testNonzero();

	void testArgwhere();

	void testFlatnonzero();

	void testWhere();

	void testSearchsorted();

	void testExtract();

	void testCount_nonzero();
}