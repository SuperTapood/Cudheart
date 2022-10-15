#pragma once

typedef int TestID;

// id structure:
// [2][2][2][1]
// [module][submodule][function][overload]
// module starts from 10 (10 - arrays, 11 - exceptions etc.)

static const TestID ARRAYS_IO_FROM_STRING_SCI = 10000;
static const TestID ARRAYS_IO_FROM_STRING_S = 10001;
static const TestID ARRAYS_IO_FROM_STRING_SI = 10002;
static const TestID ARRAYS_IO_FROM_STRING_SS = 10003;
static const TestID ARRAYS_IO_SAVE_SAC = 10010;
static const TestID ARRAYS_IO_SAVE_SA = 10011;
static const TestID ARRAYS_IO_FROM_FILE_SCI = 10020;
static const TestID ARRAYS_IO_FROM_FILE_SC = 10021;
static const TestID ARRAYS_IO_FROM_FILE_SI = 10022;
static const TestID ARRAYS_IO_FROM_FILE_S = 10023;