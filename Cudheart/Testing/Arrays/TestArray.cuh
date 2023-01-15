#pragma once
#include <stdio.h>
#include "../TestUtil.cuh"
#include "ArrayOps/ArrayOps.cuh"
#include "VecOpsTest/VecOpsTest.cuh"
#include "IO/IO.cuh"
#include "VecTest/VecTest.cuh"
#include "MatTest/MatTest.cuh"
#include "MatOpsTest/MatOpsTest.cuh"

namespace Cudheart::Testing {
	void testArrays();
}