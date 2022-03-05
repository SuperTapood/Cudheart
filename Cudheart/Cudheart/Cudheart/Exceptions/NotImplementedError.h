#pragma once

#include "BaseException.h"


class NotImplementedError : public BaseException {
public:
	NotImplementedError(string name);
};