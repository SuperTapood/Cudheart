#pragma once

typedef int TestID;

// id structure:
// [2][2][2][1]
// [module][submodule][function][overload]
// module starts from 10 (10 - arrays, 11 - exceptions etc.)

#pragma region Arrays

#pragma region ArrOps
static const TestID ARRAYS_ARROPS_APPEND_VecT = 10000;
static const TestID ARRAYS_ARROPS_APPEND_MatVec = 10001;
#pragma endregion

#pragma region IO
static const TestID ARRAYS_IO_FROM_STRING_SCI = 20000;
static const TestID ARRAYS_IO_FROM_STRING_S = 20001;
static const TestID ARRAYS_IO_FROM_STRING_SI = 20002;
static const TestID ARRAYS_IO_FROM_STRING_SS = 20003;
static const TestID ARRAYS_IO_SAVE_SAC = 20010;
static const TestID ARRAYS_IO_SAVE_SA = 20011;
static const TestID ARRAYS_IO_FROM_FILE_SCI = 20020;
static const TestID ARRAYS_IO_FROM_FILE_SC = 20021;
static const TestID ARRAYS_IO_FROM_FILE_SI = 20022;
static const TestID ARRAYS_IO_FROM_FILE_S = 20023;
static const TestID ARRAYS_IO_LOAD_SCI = 20030;
static const TestID ARRAYS_IO_LOAD_SC = 20031;
static const TestID ARRAYS_IO_LOAD_SI = 20032;
static const TestID ARRAYS_IO_LOAD_S = 20033;
static const TestID ARRAYS_IO_FROM_FUNCTION_FuncI = 20040;
#pragma endregion

#pragma endregion